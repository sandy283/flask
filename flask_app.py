# app.py
from flask import Flask, render_template, request
import pinecone
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'bc550af9-4a7e-4ad7-b50e-ce9920f122b0')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']

        if user_input == 'exit':
            return render_template('index.html', response='Exiting')

        if user_input == '':
            return render_template('index.html', response=None, source_documents=None)

        result = qa({"query": user_input})
        response = result["result"]
        source_documents = result["source_documents"]

        return render_template('index.html', response=response, source_documents=source_documents)

    return render_template('index.html', response=None, source_documents=None)

if __name__ == '__main__':
    # Initialize Pinecone and other components
    pinecone.init(api_key='bc550af9-4a7e-4ad7-b50e-ce9920f122b0', environment='gcp-starter')
    index = pinecone.Index('langchainpinecone')

    extracted_data = load_pdf_file(data='/data')
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings()
    index_name = 'langchainpinecone'
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    prompt_template = """
    Use the following pieces of information to answer the problem faced in the Continuous casting machines or centrifugal casting machines.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}
    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={'max_new_tokens': 1024,
                                'temperature': 0.3})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 1}),
                                     return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

    app.run(debug=True, host='0.0.0.0')
