import os
import streamlit as st
import os
import streamlit as st
from phi.assistant import Assistant
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.llm.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.toolkit import ToolKit
from phi.tools.function import tool_function

# --- Configuration for File Tools ---
# For security, the assistant will only be allowed to access files
# within this specific directory.
ALLOWED_FILES_DIR = "/Users/jearthur/Desktop/"

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please set it to run the assistant.")
    st.stop()

# Path to the PDF file
pdf_file_path = "/Users/jearthur/Desktop/GRL_Book.pdf" # This will be the path provided by the user

# 1. Create Knowledge Base
# Check if the PDF file exists
if not os.path.exists(pdf_file_path):
    st.error(f"PDF file not found at: {pdf_file_path}")
    st.error("Please ensure the PDF file is at the specified path and the application has read access.")
    st.stop()

knowledge_base = PDFKnowledgeBase(
    path=pdf_file_path,
    # Using default PDFReader settings, which should chunk by default.
)
# It's important to load the knowledge base.
# recreate=False means it will load from existing if available, True will rebuild.
# For a local LanceDB, loading each time might be acceptable for a single book.
# For production, you'd load it once and potentially save the DB.
# Add a check to see if the knowledge base has been loaded before trying to load again in Streamlit context
if 'knowledge_loaded' not in st.session_state:
    try:
        knowledge_base.load(recreate=True) # Forcing recreate for now to ensure freshness during dev
        st.session_state.knowledge_loaded = True
        print("Knowledge base loaded successfully.")
    except Exception as e:
        st.error(f"Error loading PDF knowledge base: {e}")
        st.stop()
elif not st.session_state.knowledge_loaded:
    # If loading failed previously, show error and stop
    st.error("Knowledge base failed to load in a previous attempt. Please check PDF and restart.")
    st.stop()


# --- Custom File Access Tools ---
class DesktopFileTools(ToolKit):
    @tool_function
    def list_files_on_desktop(self) -> str:
        """
        Lists all files and directories directly under /Users/jearthur/Desktop/.
        Use this tool to find out what files are available on the user's desktop.
        """
        if not os.path.exists(ALLOWED_FILES_DIR):
            return f"Error: The directory {ALLOWED_FILES_DIR} does not exist."
        if not os.path.isdir(ALLOWED_FILES_DIR):
            return f"Error: {ALLOWED_FILES_DIR} is not a directory."

        try:
            files = os.listdir(ALLOWED_FILES_DIR)
            if not files:
                return "The directory /Users/jearthur/Desktop/ is empty."
            return "\n".join(files)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    @tool_function
    def read_file_from_desktop(self, filename: str) -> str:
        """
        Reads the content of a specified file from the /Users/jearthur/Desktop/ directory.
        Provide the exact filename (e.g., 'notes.txt', 'script.py').
        This tool should primarily be used for text-based files. Reading binary files may produce unhelpful output.
        """
        if not isinstance(filename, str) or not filename.strip():
            return "Error: You must provide a filename."

        # Basic security: Prevent path traversal
        if ".." in filename or filename.startswith("/"):
            return "Error: Invalid filename. Path traversal is not allowed."

        file_path = os.path.join(ALLOWED_FILES_DIR, filename)

        if not os.path.exists(file_path):
            return f"Error: File '{filename}' not found in {ALLOWED_FILES_DIR}."
        if not os.path.isfile(file_path):
            return f"Error: '{filename}' is not a file."

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Limit output size to prevent overwhelming the context window
            max_length = 5000  # characters
            if len(content) > max_length:
                return content[:max_length] + f"\n\n[File content truncated to {max_length} characters]"
            if not content.strip():
                return f"The file '{filename}' is empty or contains only whitespace."
            return content
        except Exception as e:
            return f"Error reading file '{filename}': {str(e)}"

# 2. Create Vector Database (LanceDB)
# LanceDB will store embeddings in memory or locally on disk.
# By default, it creates a table in a local `./lancedb` directory.
vector_db = LanceDb(
    collection="pdf_embeddings_grl_book",
    embedder="openai", # Using OpenAI embeddings, ensure OPENAI_API_KEY is set
)

# 3. Create Assistant
assistant = Assistant(
    llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key), # Specify the LLM
    knowledge_base=knowledge_base,
    vector_db=vector_db,
    description="You are a helpful AI assistant specializing in Graph Neural Networks. Use your knowledge base (GRL_Book.pdf), web search, and file system tools to answer questions and perform tasks.",
    instructions=[
        "When answering questions, first try to find relevant information in the GRL_Book.pdf knowledge base.",
        "If the information is not found in the book, you can use the DuckDuckGo web search tool to find relevant information online.",
        "You can also list files on the user's Desktop directory (/Users/jearthur/Desktop/) using the `list_files_on_desktop` tool.",
        "You can read text-based files from the user's Desktop directory using the `read_file_from_desktop` tool. Provide the filename.",
        "If you use web search, briefly state that you are searching online or that the information comes from a web search.",
        "If you use file system tools, state which tool you are using and what you are doing.",
        "If information cannot be found in the book, via web search, or relevant files, state that.",
        "Be precise and refer to the knowledge base, web search results, or file contents when possible.",
    ],
    tools=[DuckDuckGo(), DesktopFileTools()],
    show_tool_calls=True, # Good for debugging
    search_knowledge=True, # Enable searching the knowledge base
    read_chat_history=True, # Enable reading chat history for context
)

st.title("📘 GNN PDF, Web & Desktop Assistant")
st.caption("Ask questions about the GRL_Book.pdf")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask something about Graph Neural Networks..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            response_stream = assistant.run(prompt)
            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I encountered an error."
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.header("About")
st.sidebar.info(
    "This is an AI assistant that can answer questions about the content of 'GRL_Book.pdf' "
    "on the topic of Graph Neural Networks. "
    "It uses the phidata library with an OpenAI LLM and LanceDB for vector storage."
)
st.sidebar.info(f"Knowledge Base: {pdf_file_path}")

# For running this app:
# 1. Save this code as pdf_assistant_app.py
# 2. Set your OPENAI_API_KEY: export OPENAI_API_KEY='your_key_here'
# 3. Install dependencies: pip install phidata streamlit openai lancedb pypdf duckduckgo-search
# 4. Run: streamlit run pdf_assistant_app.py
# Ensure GRL_Book.pdf is at /Users/jearthur/Desktop/GRL_Book.pdf
# or change the pdf_file_path variable.
# The first run might take a moment to process and embed the PDF.
print("PDF Assistant App initialized. Ready for Streamlit to run.")
