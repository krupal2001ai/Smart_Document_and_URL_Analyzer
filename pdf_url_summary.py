import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import sys
import asyncio
import validators
from urllib.parse import urlparse, parse_qs
from langchain_community.document_loaders import YoutubeLoader, PlaywrightURLLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
import re

import subprocess

try:
    subprocess.run(["playwright", "install", "chromium"], check=True)
except Exception as e:
    print("Playwright installation may have already been done or failed:", e)

# Fix Windows compatibility
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gc
import psutil

# Add this function to clean up memory
def cleanup_memory():
    """Clean up memory and temporary files"""
    gc.collect()
    
    # Clear large session state items if needed
    if 'vectordb' in st.session_state and st.session_state.vectordb:
        del st.session_state.vectordb
        st.session_state.vectordb = None
    
    gc.collect()

def show_memory_usage():
    """Display current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    st.sidebar.metric("Memory Usage", f"{memory_mb:.1f} MB")

# Add cleanup button in sidebar
if st.sidebar.button(" Clear Memory"):
    cleanup_memory()
    st.sidebar.success("Memory cleared!")
    st.rerun()
# Enhanced page configuration
st.set_page_config(
    page_title="Smart Document & URL Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
            
<div class="main-header">
    <h1>Smart Document & URL Analyzer</h1>
    <p>AI-Based Summarization of PDFs, YouTube videos, and websites</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### Configuration")
    
    # Input type selection
    input_type = st.radio(
        "Choose your input type:",
        options=[" PDF Documents", " URLs & YouTube"],
        index=0,
        help="Select whether you want to analyze PDF documents or web content"
    )
    
    st.markdown("---")
    
    # API Keys section
    st.markdown("### API Configuration")
    groq_api_key = st.text_input(
        "Groq API Key", 
        type='password',
        help="Get your free API key from console.groq.com"
    )
    
    if "PDF" in input_type:
        cohere_api_key = st.text_input(
            "Cohere API Key", 
            type='password',
            help="Required for PDF embeddings - get from cohere.ai"
        )
    
    # Model configuration
    if groq_api_key:
        st.markdown("### Model Settings")
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7,
            help="Higher values make output more creative, lower values more focused"
        )
        
        model_options = [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "qwen-qwq-32b",
            "mistral-saba-24b"
        ]
        
        model = st.selectbox(
            "Model:", 
            model_options,
            help="Choose the AI model for processing"
        )

# Prompt templates for different summary types
PROMPT_TEMPLATES = {
    "Point-wise Summary": {
        "map": "Create a detailed bullet-point summary of the following content:\n\n{text}",
        "combine": "Combine these bullet points into a comprehensive point-wise summary:\n\n{text}"
    },
    " Quick Summary": {
        "map": "Create a brief, concise summary of the following content in 2-3 sentences:\n\n{text}",
        "combine": "Combine these summaries into a short, cohesive overview:\n\n{text}"
    },
    "Detailed Analysis": {
        "map": "Provide a comprehensive analysis of the following content, including key insights and details:\n\n{text}",
        "combine": "Create a detailed, analytical summary combining all the insights:\n\n{text}"
    },
    " Language Translation": {
        "map": "Summarize the following content and translate it to the specified language:\n\n{text}",
        "combine": "Combine these translated summaries into a cohesive summary in the target language:\n\n{text}"
    },
    " Easy to Understand": {
        "map": "Explain the following content in simple, easy-to-understand language suitable for beginners:\n\n{text}",
        "combine": "Combine these explanations into a simple, beginner-friendly summary:\n\n{text}"
    }
}

def extract_youtube_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id, preferred_languages=['en', 'hi', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']):
    """Get transcript from YouTube video using youtube_transcript_api with language flexibility"""
    try:
        # Method 1: Try direct transcript retrieval with preferred languages
        for lang_code in preferred_languages:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
                transcript_text = ' '.join([item['text'] for item in transcript_data])
                st.success(f" Found transcript in: {lang_code}")
                return transcript_text, lang_code
            except Exception:
                continue
        
        # Method 2: If preferred languages fail, try to get any available transcript
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Show available languages to user
            available_languages = []
            for transcript in transcript_list:
                lang_type = "Manual" if not transcript.is_generated else "Auto-generated"
                available_languages.append(f"{transcript.language} ({transcript.language_code}) - {lang_type}")
            
            st.info("Available transcript languages:")
            for lang in available_languages:
                st.write(f"• {lang}")
            
            # Try to get the first available transcript
            for transcript in transcript_list:
                try:
                    transcript_data = transcript.fetch()
                    transcript_text = ' '.join([item['text'] for item in transcript_data])
                    found_language = transcript.language_code
                    lang_type = "Manual" if not transcript.is_generated else "Auto-generated"
                    st.success(f" Using {lang_type.lower()} transcript in: {found_language}")
                    return transcript_text, found_language
                except Exception as e:
                    st.warning(f" Failed to fetch transcript in {transcript.language_code}: {str(e)}")
                    continue
        
        except Exception as e:
            st.error(f"Error listing transcripts: {str(e)}")
        
        # Method 3: Last resort - try without language specification
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([item['text'] for item in transcript_data])
            st.success(" Retrieved transcript (language auto-detected)")
            return transcript_text, "auto-detected"
        except Exception as e:
            st.error(f" Final attempt failed: {str(e)}")
        
        st.error("No transcripts could be retrieved for this video")
        st.info(" This might be because:")
        st.info("• The video doesn't have captions enabled")
        st.info("• The video is private or restricted")
        st.info("• The video is too new (captions not yet generated)")
        st.info("• The video creator disabled captions")
        
        return None, None
            
    except Exception as e:
        st.error(f" Unexpected error: {str(e)}")
        return None, None

def clean_youtube_url(url):
    """Standardize YouTube URL format"""
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return f"https://www.youtube.com/watch?v={parsed.path.lstrip('/')}"
    elif "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        video_id = query.get('v', [''])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return url

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'conversation_rag_chain' not in st.session_state:
    st.session_state.conversation_rag_chain = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session history for conversations"""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def initialize_embeddings(api_key):
    """Initialize Cohere embeddings for document processing"""
    try:
        embeddings = CohereEmbeddings(
            cohere_api_key=api_key,
            model="embed-english-light-v3.0"
        )
        return embeddings
    except Exception as e:
        st.error(f" Error initializing embeddings: {str(e)}")
        return None

def setup_rag_chain(llm, retriever):
    """Setup the RAG chain with conversation history"""
    # Create context-aware question prompt
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "create a standalone question that can be understood "
        "without the chat history. Don't answer the question, "
        "just reformulate it if needed."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create history-aware retriever
    history_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create QA prompt
    system_prompt = (
        "You are a helpful AI assistant for question-answering tasks. "
        "Use the retrieved context to provide accurate, detailed answers. "
        "If you don't know something, be honest about it. "
        "Structure your responses clearly and provide examples when helpful."
        "\n\nContext:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create the chains
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, qa_chain)
    
    # Create conversation chain with history
    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversation_rag_chain

# Main application logic
if "PDF" in input_type:
    # PDF Processing Section
    st.markdown("## PDF Document Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4> Features</h4>
            <ul>
                <li>Upload multiple PDF documents</li>
                <li>Interactive conversation with your documents</li>
                <li>AI-powered question answering</li>
                <li>Dynamic Conversation history tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if groq_api_key and 'cohere_api_key' in locals():
            session_id = st.text_input(" Session ID", value="Session_1")
    
    if groq_api_key and 'cohere_api_key' in locals() and cohere_api_key:
        # Initialize LLM and embeddings
        llm = ChatGroq(
            model=model,
            api_key=groq_api_key,
            temperature=temperature
        )
        
        embeddings = initialize_embeddings(cohere_api_key)
        
        if embeddings:
            st.success("AI models initialized successfully!")
            
            # File upload section
            uploaded_files = st.file_uploader(
                " Upload your PDF documents",
                type="pdf",
                accept_multiple_files=True,
                help="You can upload multiple PDF files at once"
            )
            
            if uploaded_files:
                with st.spinner(" Processing your documents..."):
                    documents = []
                    processed_files = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                            documents.extend(docs)
                            processed_files.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            os.unlink(tmp_path)
                    
                    # Process documents if any were loaded successfully
                    if documents:
                        try:
                            # Split documents into chunks
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1500,
                                chunk_overlap=200
                            )
                            splits = splitter.split_documents(documents)
                            
                            # Create vector database
                            st.session_state.vectordb = FAISS.from_documents(
                                documents=splits,
                                embedding=embeddings
                            )
                            
                            # Create retriever
                            retriever = st.session_state.vectordb.as_retriever(
                                search_kwargs={"k": 5}
                            )
                            
                            # Setup RAG chain
                            st.session_state.conversation_rag_chain = setup_rag_chain(llm, retriever)
                            
                            st.markdown(f"""
                            <div class="success-message">
                                Successfully processed {len(processed_files)} files with {len(splits)} chunks!<br>
                                Files: {', '.join(processed_files)}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f" Error creating vector database: {str(e)}")
            
            # Chat interface
            if st.session_state.conversation_rag_chain:
                st.markdown("### Chat with Your Documents")
                
                # Display chat history
                if session_id in st.session_state.store:
                    chat_history = st.session_state.store[session_id]
                    for message in chat_history.messages:
                        if hasattr(message, 'content'):
                            if message.type == "human":
                                with st.chat_message("user"):
                                    st.write(message.content)
                            else:
                                with st.chat_message("assistant"):
                                    st.write(message.content)
                
                # Chat input
                user_input = st.chat_input(" Ask anything about your documents...")
                
                if user_input:
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    try:
                        with st.chat_message("assistant"):
                            with st.spinner(" Analyzing your documents..."):
                                response = st.session_state.conversation_rag_chain.invoke(
                                    {"input": user_input},
                                    config={"configurable": {"session_id": session_id}}
                                )
                                st.write(response['answer'])
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
            else:
                st.info(" Please upload PDF files to start the conversation!")
        else:
            st.error("Failed to initialize embeddings. Please check your Cohere API key.")
    else:
        st.warning(" Please enter both Groq and Cohere API keys to continue.")

else:
    # URL Processing Section
    st.markdown("##  URL & YouTube Analysis")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>What you can analyze:</h4>
            <ul>
                <li> YouTube videos (with automatic transcript extraction)</li>
                <li> Websites and blogs</li>
                <li> News articles</li>
                <li> Online documents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Summary type selection
        summary_type = st.selectbox(
            "Summary Style:",
            list(PROMPT_TEMPLATES.keys()),
            help="Choose how you want the content to be summarized"
        )
        
        # Language preference for YouTube transcripts
        st.markdown("** Transcript Language Preference:**")
        preferred_languages = st.multiselect(
            "Select preferred languages (in order of preference):",
            options=[
                'en (English)', 'hi (Hindi)', 'es (Spanish)', 'fr (French)', 
                'de (German)', 'it (Italian)', 'pt (Portuguese)', 'ru (Russian)',
                'ja (Japanese)', 'ko (Korean)', 'zh (Chinese)', 'ar (Arabic)',
                'bn (Bengali)', 'ur (Urdu)', 'ta (Tamil)', 'te (Telugu)',
                'mr (Marathi)', 'gu (Gujarati)', 'kn (Kannada)', 'ml (Malayalam)'
            ],
            default=['en (English)', 'hi (Hindi)'],
            help="App will try languages in the order you select them"
        )
        
        # Extract language codes
        lang_codes = [lang.split(' ')[0] for lang in preferred_languages] if preferred_languages else ['en', 'hi']
        
        # Language selection for translation
        target_language = None
        if "Translation" in summary_type:
            target_language = st.text_input(
                "Target Language:",
                value="Spanish",
                help="Enter the language you want the summary translated to"
            )
    
    # URL input
    url_input = st.text_input(
        "Enter URL:",
        placeholder="https://youtube.com/watch?v=... or https://example.com",
        help="Paste any YouTube video URL or website URL here"
    )
    
    # Process button
    if st.button(" Analyze Content", type="primary"):
        if not url_input:
            st.error(" Please enter a URL")
            st.stop()
        
        if not validators.url(url_input):
            st.error(" Please enter a valid URL")
            st.stop()
        
        if not groq_api_key:
            st.error(" Please enter your Groq API key")
            st.stop()
        
        try:
            # Initialize the language model
            llm = ChatGroq(api_key=groq_api_key, model=model, temperature=temperature)
            
            # Process different types of URLs
            if "youtube.com" in url_input or "youtu.be" in url_input:
                st.info(" Processing YouTube video...")
                
                # Extract video ID and get transcript
                video_id = extract_youtube_id(url_input)
                if video_id:
                    transcript, found_language = get_youtube_transcript(video_id, lang_codes)
                    
                    if transcript:
                        docs = [Document(page_content=transcript)]
                        
                        # Add language info to the summary context
                        if found_language and found_language != 'en':
                            st.info(f" Note: Transcript is in {found_language}. The AI will process it accordingly.")
                            
                        # Modify prompts to handle non-English content
                        if found_language and found_language != 'en':
                            language_context = f"\n\nNote: The original content is in {found_language}. Please provide the summary in English unless otherwise specified."
                            selected_prompts = PROMPT_TEMPLATES[summary_type].copy()
                            selected_prompts["map"] += language_context
                            selected_prompts["combine"] += language_context
                        else:
                            selected_prompts = PROMPT_TEMPLATES[summary_type]
                            
                    else:
                        st.error("Could not extract transcript from this video")
                        st.info(" Try selecting different language preferences or check if the video has captions enabled")
                        st.stop()
                else:
                    st.error("Could not extract video ID from URL")
                    st.stop()
            else:
                st.info("Scraping website content...")
                try:
                    loader = PlaywrightURLLoader(
                        urls=[url_input], 
                        remove_selectors=["header", "footer", "nav", ".advertisement", ".ads"]
                    )
                    docs = loader.load()
                    
                    if docs and docs[0].page_content.strip():
                        st.success(" Website content extracted successfully!")
                    else:
                        st.error(" No content could be extracted from this website")
                        st.stop()
                except Exception as e:
                    st.error(f"Error scraping website: {str(e)}")
                    st.stop()
            
            # Create prompts based on user selection
            if 'selected_prompts' not in locals():
                selected_prompts = PROMPT_TEMPLATES[summary_type]
            
            # Modify prompts for language translation
            if "Translation" in summary_type and target_language:
                selected_prompts["map"] += f" Translate the summary to {target_language}."
                selected_prompts["combine"] += f" Ensure the final summary is in {target_language}."
            
            map_prompt = PromptTemplate(
                input_variables=["text"],
                template=selected_prompts["map"]
            )
            
            combine_prompt = PromptTemplate(
                input_variables=["text"],
                template=selected_prompts["combine"]
            )
            
            # Create summarization chain
            chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=True
            )
            
            # Generate summary
            with st.spinner(" Generating your summary..."):
                summary = chain.run(docs)
            
            # Display results
            st.markdown("###  Summary Results")
            st.markdown("---")
            
            # Show summary in a nice format
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                <h4 style="color: #667eea; margin-bottom: 1rem;"> {summary_type}</h4>
                <div style="line-height: 1.6;">
                    {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional options
            st.markdown("###  Additional Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(" Generate Different Style"):
                    st.rerun()
            
            with col2:
                if st.button(" Save Summary"):
                    st.download_button(
                        label=" Download Summary",
                        data=summary,
                        file_name=f"summary_{summary_type.replace(' ', '_').lower()}.txt",
                        mime="text/plain"
                    )
            
            with col3:
                if st.button(" Analyze Another URL"):
                    st.rerun()
                    
        except Exception as e:
            st.error("An error occurred while processing")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Smart Document & URL Analyzer - Powered by Krupal Varama</p>
    <p>Built with using Streamlit, LangChain, and Groq, Cohere</p>
</div>
""", 
unsafe_allow_html=True)