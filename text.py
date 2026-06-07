import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import uuid

# Setting up the LLM model
llm = Ollama(model="llama2")

# Create the temporary directory if it does not exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Defining the PDF GPT class
class PdfGpt():
    def __init__(self, uploaded_file):
        # Generate a unique file name for the uploaded PDF
        file_id = str(uuid.uuid4())
        file_path = os.path.join("temp", f"{file_id}.pdf")

        # Save the uploaded PDF to a temporary location
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Split the PDF into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents=PyMuPDFLoader(file_path=file_path).load())
        
        # Initialize the embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device':'cpu'},
            encode_kwargs = { 'normalize_embeddings': True }
        )
        
        # Create a vector store for the chunks
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("vectorstore")
        
        # Define the template for the Prompt
        template = """
        ### System:
        You are a respectful and honest assistant. You have to answer the user's questions using only the context \
        provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

        ### Context:
        {context}

        ### User:
        {question}

        ### Response:
        """

        # Initialize the Retrieval QA model
        self.hey = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            return_source_documents=True, 
            chain_type_kwargs={'prompt': PromptTemplate.from_template(template) } 
        )

# Set the Streamlit app title
st.title("PDF GPT")

# Add futuristic CSS styling
st.markdown(
    """
    <style>
    body {
        color: #ffffff;
        background-color: #121212;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #03dac6;
        border-radius: 12px;
        font-family: 'Arial', sans-serif;
        color: #121212;
    }
    .stFileUploader>div>div>div {
        background-color: #03dac6;
        border-radius: 12px;
        font-family: 'Arial', sans-serif;
        color: #121212;
    }
    .stTextInput>div>div>div>input {
        background-color: #03dac6;
        border-radius: 12px;
        font-family: 'Arial', sans-serif;
        color: #121212;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# File upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], help="Please upload a PDF file.", key="file_uploader", accept_multiple_files=False)

# If a file is uploaded
if uploaded_file is not None:
    # Processing the PDF
    with st.spinner("Processing PDF..."):
        oracle = PdfGpt(uploaded_file)

    # User input section
    ask = st.text_input("What's up?", key="ask", label_visibility='hidden')

    # If the user input is not empty
    if ask not in [None, "", []]:  
        # Displaying context and user input
        st.caption("📓")
        
        # Displaying prediction with expander
        with st.expander("🦙"):
            st.markdown( llm.predict(ask) )
        
        # Generating response
        with st.spinner("Generating Response..."):
            response = oracle.hey({'query': ask})
        
        # Displaying response
        st.markdown( response['result'] )
