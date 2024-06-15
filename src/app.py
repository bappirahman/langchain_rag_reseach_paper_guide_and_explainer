import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_emb = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def load_vector():
    """
    Load the vector embeddings for the documents.

    This function checks if the 'vector' key is present in the session state. If not, it loads the documents from the
    given queries using the ArxivLoader class. The documents are then split into chunks using the RecursiveCharacterTextSplitter
    class with a chunk size of 1000 characters and a chunk overlap of 0. Finally, the vector embeddings are computed using
    the FAISS class with the help of the hf_emb object.

    Parameters:
    None

    Returns:
    None
    """
    if 'vector' not in st.session_state:
      start = time.time()
      print(start)
      queries = ['2302.13971', '2307.09288']
      documents = []
      
      for query in queries:
          loader = ArxivLoader(query=query, load_max_docs=1)
          documents.extend(loader.load())
      st.session_state.documents = documents
      st.session_state.docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(st.session_state.documents)
      print('Embedding started') 
      embedding_start = time.time()
      st.session_state.vector = FAISS.from_documents(st.session_state.docs, hf_emb)
      print('Embedding done')
      print('Embedding done in', (time.time() - embedding_start) / 60, 'minutes')
      st.write('It took {time.time() - start} to load the data')

chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192"
) 

prompt_template = ChatPromptTemplate.from_messages([
  """
  You are a helpful assistant who will answer questions based on the context provided. If you don't know the answer, just say that you don't know. DO NOT make up an answer.
  <context>
  {context}
  </context>
  <question>
  Question: {input}
  </question>
  """
])
if st.button('Load Data'):
  st.write('Please wait while we load the data...')
  load_vector()
  st.write('Data loaded successfully!')
input = st.text_input("Enter Your question about llama, llama2 from it's research papers")

if input:
  chain = create_stuff_documents_chain(
    prompt=prompt_template,
    llm=chat
  )
  retriever = st.session_state.vector.as_retriever()
  retriever_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=chain
  )
  response = retriever_chain.invoke({"input": input})
  st.write(response["answer"])




