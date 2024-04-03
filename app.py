from dotenv import load_dotenv
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers, HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate

from flask import Flask, render_template, jsonify, request
from src.constants import *
from src.helper import *

app = Flask(__name__)

# Get the embeddings
embeddings1 = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', 
                                    model_kwargs = {'device': "cpu"})

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

def preprocess_text(documents):
    for i in range(len(documents)):
        junk_text = f"{TEXT1}{i+1}{TEXT2}"
        documents[i].page_content = documents[i].page_content.replace(junk_text, "")
    
    documents = documents[13:]

    return documents

def train_model():
    #Load the content from PDF document
    document_loader = PyPDFDirectoryLoader('./data')
    documents = document_loader.load()
    print(len(documents))
    documents = preprocess_text(documents)
    print(len(documents))
    print(documents[1].page_content)

    #Create Chunks from documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents=documents)

    # Convert Text Chunks to embeddings and store in Vector DB
    vector_store = FAISS.from_documents(text_chunks, embeddings1)
    vector_store.save_local('./db/vector_store')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def query_llm(query):
    # Load the model
    llm = CTransformers(model='./model/llama-2-7b-chat.ggmlv3.q4_0.bin', 
                        config={'temperature':0.01, 'max_new_tokens':128},
                        model_type='llama')
    
    llm_hug = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
                             huggingfacehub_api_token=HUGGINGFACE_API_KEY, 
                             model_kwargs={"temperature":0.8, "max_length":1000})
    
    #load cector_store from local
    vector_store = FAISS.load_local('./db/vector_store', embeddings1)

    # Create Prompt Template
    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

    print("Calling chain")
    # Create the RetrievalQA Chain
    qa = RetrievalQA.from_chain_type(llm=llm_hug, 
                                    retriever=vector_store.as_retriever(search_type='similarity',
                                                                         search_kwargs={'k':3}),
                                    chain_type_kwargs={'prompt':prompt},
                                    return_source_documents=True)
    
    #print(qa({'query': "Can you let me know about WO Delay functionality in ITSM?"}))
    
    result = qa({'query': query})

    print(f"Output: {result}")

    return result['result']

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"input: {input}")
    result = query_llm(input)
    print(f"Answer: {result}")
    return str(result)

if __name__ == '__main__':
    #train_model()
    #query_llm()
    app.run(host='0.0.0.0', port=8090, debug=False)