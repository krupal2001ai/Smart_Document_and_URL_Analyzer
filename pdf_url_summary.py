
# import streamlit as st
# import tempfile
# import os
# import re
# import gc
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_cohere import CohereEmbeddings
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import YoutubeLoader
# from langchain.docstore.document import Document
# from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# # --- Session State Initialization ---
# if 'store' not in st.session_state:
#     st.session_state.store = {}
# if 'vectordb' not in st.session_state:
#     st.session_state.vectordb = None
# if 'conversation_rag_chain' not in st.session_state:
#     st.session_state.conversation_rag_chain = None

# # --- Utility Functions ---
# def get_session_history(session_id: str) -> ChatMessageHistory:
#     """Manage chat history for sessions."""
#     if session_id not in st.session_state.store:
#         st.session_state.store[session_id] = ChatMessageHistory()
#     return st.session_state.store[session_id]

# def initialize_embeddings(api_key):
#     """Initialize Cohere embeddings."""
#     try:
#         return CohereEmbeddings(cohere_api_key=api_key, model="embed-english-light-v3.0")
#     except Exception as e:
#         st.error(f"Error initializing embeddings: {str(e)}")
#         return None

# def setup_rag_chain(llm, retriever):
#     """Setup Retrieval-Augmented Generation chain with history."""
#     contextualize_q_system_prompt = (
#         "Given the chat history and the latest user question, "
#         "create a standalone question that can be understood without the chat history."
#     )
#     contextualize_q_prompt = ChatPromptTemplate.from_messages([
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ])
    
#     history_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
#     system_prompt = (
#         "You are an AI assistant for question-answering. "
#         "Use the retrieved context to provide accurate, clear answers. "
#         "If unsure, say so.\n\nContext:\n{context}"
#     )
#     qa_prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ])
    
#     qa_chain = create_stuff_documents_chain(llm, qa_prompt)
#     rag_chain = create_retrieval_chain(history_retriever, qa_chain)
    
#     return RunnableWithMessageHistory(
#         rag_chain,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="answer"
#     )

# def extract_youtube_id(url):
#     """Extract YouTube video ID from URL."""
#     patterns = [
#         r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
#         r'youtube\.com/watch\?.*v=([^&\n?#]+)'
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, url)
#         if match:
#             return match.group(1)
#     return None

# def get_youtube_transcript(video_id):
#     """Get YouTube transcript with user-friendly error handling."""
#     try:
#         transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         transcript_text = ' '.join([item['text'] for item in transcript_data])
#         return transcript_text
#     except (NoTranscriptFound, TranscriptsDisabled):
#         st.warning("This YouTube video does not have captions available. "
#                   "Captions may be disabled, not yet generated, or the video is private/restricted.")
#         return None
#     except Exception as e:
#         st.warning(f"Unable to retrieve captions for this video: {str(e)}")
#         return None

# def cleanup_memory():
#     """Clean up memory and resources."""
#     gc.collect()
#     if 'vectordb' in st.session_state and st.session_state.vectordb:
#         del st.session_state.vectordb
#         st.session_state.vectordb = None
#     gc.collect()

# def clear_selected_session(session_id):
#     """Clear a specific session ID from the store."""
#     if session_id in st.session_state.store:
#         del st.session_state.store[session_id]
#         gc.collect()
#         st.success(f"Session {session_id} cleared successfully!")

# # --- Main Application ---
# st.title("Document & URL Chatbot")

# # Sidebar Configuration
# with st.sidebar:
#     st.subheader("Configuration")
#     input_type = st.radio("Input Type:", ["PDF Documents", "URLs"])
#     groq_api_key = st.text_input("Groq API Key", type='password')
#     cohere_api_key = st.text_input("Cohere API Key", type='password')
    
#     if groq_api_key:
#         model = st.selectbox("Model:", [
#             "llama-3.3-70b-versatile",
#             "mixtral-8x7b-32768",
#             "gemma2-9b-it"
#         ])
#         temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
#     # Session Management
#     st.subheader("Manage Sessions")
#     if st.session_state.store:
#         session_to_clear = st.selectbox("Select Session to Clear:", [""] + list(st.session_state.store.keys()))
#         if session_to_clear and st.button("Clear Selected Session"):
#             clear_selected_session(session_to_clear)
#             st.rerun()
    
#     if st.button("Clear All Memory"):
#         cleanup_memory()
#         st.session_state.store.clear()
#         st.success("All memory and sessions cleared!")

# # Initialize LLM and Embeddings
# if groq_api_key and cohere_api_key:
#     llm = ChatGroq(model=model, api_key=groq_api_key, temperature=temperature)
#     embeddings = initialize_embeddings(cohere_api_key)
    
#     if embeddings:
#         # Input Processing
#         documents = []
        
#         if input_type == "PDF Documents":
#             st.subheader("Upload PDFs")
#             uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
            
#             if uploaded_files:
#                 with st.spinner("Processing PDFs..."):
#                     for uploaded_file in uploaded_files:
#                         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#                             tmp_file.write(uploaded_file.getvalue())
#                             tmp_file_path = tmp_file.name
                        
#                         try:
#                             loader = PyPDFLoader(tmp_file_path)
#                             documents.extend(loader.load())
#                         except Exception as e:
#                             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#                         finally:
#                             try:
#                                 loader = None
#                                 gc.collect()
#                                 os.unlink(tmp_file_path)
#                             except PermissionError:
#                                 st.warning(f"Could not delete {tmp_file_path}. It will be cleaned up later.")
#                             except Exception as e:
#                                 st.error(f"Error deleting {tmp_file_path}: {str(e)}")
        
#         else:
#             st.subheader("Enter URL")
#             url_input = st.text_input("Enter URL (YouTube or Website):")
#             if url_input and st.button("Process URL"):
#                 with st.spinner("Processing URL..."):
#                     video_id = extract_youtube_id(url_input)
#                     if video_id:
#                         transcript = get_youtube_transcript(video_id)
#                         if transcript:
#                             documents = [Document(page_content=transcript)]
#                         else:
#                             st.stop()
#                     else:
#                         try:
#                             loader = WebBaseLoader(
#                                 web_paths=[url_input],
#                                 requests_kwargs={"verify": True, "timeout": 30}
#                             )
#                             documents = loader.load()
#                             if not documents or not documents[0].page_content.strip():
#                                 st.warning("No content could be extracted from this website. "
#                                          "The page may be dynamically loaded, restricted, or empty.")
#                                 st.stop()
#                             else:
#                                 st.success("Website content extracted successfully!")
#                         except Exception as e:
#                             st.warning(f"Unable to process website content: {str(e)}. "
#                                       "Try a different URL or check if the website is accessible.")
#                             st.stop()
        
#         # Process Documents and Setup RAG
#         if documents:
#             splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
#             splits = splitter.split_documents(documents)
#             st.session_state.vectordb = FAISS.from_documents(splits, embeddings)
#             retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 5})
#             st.session_state.conversation_rag_chain = setup_rag_chain(llm, retriever)
#             st.success("Content processed successfully!")
        
#         # Chatbot Interface
#         if st.session_state.conversation_rag_chain:
#             st.subheader("Chat with Content")
#             session_id = st.text_input("Session ID", value="Session_1")
            
#             # Display Chat History
#             if session_id in st.session_state.store:
#                 for message in st.session_state.store[session_id].messages:
#                     with st.chat_message(message.type):
#                         st.write(message.content)
            
#             # User Input
#             user_input = st.chat_input("Ask a question about the content...")
#             if user_input:
#                 with st.chat_message("user"):
#                     st.write(user_input)
#                 with st.chat_message("assistant"):
#                     with st.spinner("Generating response..."):
#                         try:
#                             response = st.session_state.conversation_rag_chain.invoke(
#                                 {"input": user_input},
#                                 config={"configurable": {"session_id": session_id}}
#                             )
#                             st.write(response['answer'])
#                         except Exception as e:
#                             st.error(f"Error generating response: {str(e)}")
#     else:
#         st.error("Failed to initialize embeddings. Check Cohere API key.")
# else:
#     st.warning("Please enter both Groq and Cohere API keys.")

import streamlit as st
import tempfile
import os
import re
import gc
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
# Corrected import for get_transcript
from youtube_transcript_api import get_transcript, NoTranscriptFound, TranscriptsDisabled

# --- Session State Initialization ---
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'conversation_rag_chain' not in st.session_state:
    st.session_state.conversation_rag_chain = None

# --- Utility Functions ---
def get_session_history(session_id: str) -> ChatMessageHistory:
    """Manage chat history for sessions."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def initialize_embeddings(api_key):
    """Initialize Cohere embeddings."""
    try:
        return CohereEmbeddings(cohere_api_key=api_key, model="embed-english-light-v3.0")
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def setup_rag_chain(llm, retriever):
    """Setup Retrieval-Augmented Generation chain with history."""
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "create a standalone question that can be understood without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    system_prompt = (
        "You are an AI assistant for question-answering. "
        "Use the retrieved context to provide accurate, clear answers. "
        "If unsure, say so.\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, qa_chain)
    
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

def extract_youtube_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """Get YouTube transcript with user-friendly error handling."""
    try:
        # Corrected function call
        transcript_data = get_transcript(video_id, languages=['en'])
        transcript_text = ' '.join([item['text'] for item in transcript_data])
        return transcript_text
    except (NoTranscriptFound, TranscriptsDisabled):
        st.warning("This YouTube video does not have captions available. "
                   "Captions may be disabled, not yet generated, or the video is private/restricted.")
        return None
    except Exception as e:
        st.warning(f"Unable to retrieve captions for this video: {str(e)}")
        return None

def cleanup_memory():
    """Clean up memory and resources."""
    gc.collect()
    if 'vectordb' in st.session_state and st.session_state.vectordb:
        del st.session_state.vectordb
        st.session_state.vectordb = None
    gc.collect()

def clear_selected_session(session_id):
    """Clear a specific session ID from the store."""
    if session_id in st.session_state.store:
        del st.session_state.store[session_id]
        gc.collect()
        st.success(f"Session {session_id} cleared successfully!")

# --- Main Application ---
st.title("Document & URL Chatbot")

# Sidebar Configuration
with st.sidebar:
    st.subheader("Configuration")
    input_type = st.radio("Input Type:", ["PDF Documents", "URLs"])
    groq_api_key = st.text_input("Groq API Key", type='password')
    cohere_api_key = st.text_input("Cohere API Key", type='password')
    
    if groq_api_key:
        model = st.selectbox("Model:", [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Session Management
    st.subheader("Manage Sessions")
    if st.session_state.store:
        session_to_clear = st.selectbox("Select Session to Clear:", [""] + list(st.session_state.store.keys()))
        if session_to_clear and st.button("Clear Selected Session"):
            clear_selected_session(session_to_clear)
            st.rerun()
    
    if st.button("Clear All Memory"):
        cleanup_memory()
        st.session_state.store.clear()
        st.success("All memory and sessions cleared!")

# Initialize LLM and Embeddings
if groq_api_key and cohere_api_key:
    llm = ChatGroq(model=model, api_key=groq_api_key, temperature=temperature)
    embeddings = initialize_embeddings(cohere_api_key)
    
    if embeddings:
        # Input Processing
        documents = []
        
        if input_type == "PDF Documents":
            st.subheader("Upload PDFs")
            uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
            
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            loader = PyPDFLoader(tmp_file_path)
                            documents.extend(loader.load())
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            try:
                                loader = None
                                gc.collect()
                                os.unlink(tmp_file_path)
                            except PermissionError:
                                st.warning(f"Could not delete {tmp_file_path}. It will be cleaned up later.")
                            except Exception as e:
                                st.error(f"Error deleting {tmp_file_path}: {str(e)}")
        
        else:
            st.subheader("Enter URL")
            url_input = st.text_input("Enter URL (YouTube or Website):")
            if url_input and st.button("Process URL"):
                with st.spinner("Processing URL..."):
                    video_id = extract_youtube_id(url_input)
                    if video_id:
                        transcript = get_youtube_transcript(video_id)
                        if transcript:
                            documents = [Document(page_content=transcript)]
                        else:
                            st.stop()
                    else:
                        try:
                            loader = WebBaseLoader(
                                web_paths=[url_input],
                                requests_kwargs={"verify": True, "timeout": 30}
                            )
                            documents = loader.load()
                            if not documents or not documents[0].page_content.strip():
                                st.warning("No content could be extracted from this website. "
                                           "The page may be dynamically loaded, restricted, or empty.")
                                st.stop()
                            else:
                                st.success("Website content extracted successfully!")
                        except Exception as e:
                            st.warning(f"Unable to process website content: {str(e)}. "
                                       "Try a different URL or check if the website is accessible.")
                            st.stop()
        
        # Process Documents and Setup RAG
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            splits = splitter.split_documents(documents)
            st.session_state.vectordb = FAISS.from_documents(splits, embeddings)
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 5})
            st.session_state.conversation_rag_chain = setup_rag_chain(llm, retriever)
            st.success("Content processed successfully!")
        
        # Chatbot Interface
        if st.session_state.conversation_rag_chain:
            st.subheader("Chat with Content")
            session_id = st.text_input("Session ID", value="Session_1")
            
            # Display Chat History
            if session_id in st.session_state.store:
                for message in st.session_state.store[session_id].messages:
                    with st.chat_message(message.type):
                        st.write(message.content)
            
            # User Input
            user_input = st.chat_input("Ask a question about the content...")
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        try:
                            response = st.session_state.conversation_rag_chain.invoke(
                                {"input": user_input},
                                config={"configurable": {"session_id": session_id}}
                            )
                            st.write(response['answer'])
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
    else:
        st.error("Failed to initialize embeddings. Check Cohere API key.")
else:
    st.warning("Please enter both Groq and Cohere API keys.")


    

