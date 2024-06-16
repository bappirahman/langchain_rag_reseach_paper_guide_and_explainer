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

st.title('Research Paper Guide and Explainer')

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_emb = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def load_vector(query):
    """
    Load the data from Arxviv and store the vector embeddings.

    This function checks if the 'vector' key is present in the session state. If not, it loads the documents from the given queries using the ArxivLoader class. The documents are then split into chunks using the RecursiveCharacterTextSplitter class with a chunk size of 1000 characters and a chunk overlap of 100. Finally, the vector embeddings are computed using the FAISS class with the help of the hf_emb object.

    Parameters:
    - query (str): The query string used to load the documents.

    Returns:
    None
    """
    if 'vector' not in st.session_state:
      start = time.time()
      print(start)
      
      st.session_state.documents = ArxivLoader(query=query, load_max_docs=1).load()
      st.session_state.docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(st.session_state.documents)
      print('Embedding started') 
      embedding_start = time.time()
      st.session_state.vector = FAISS.from_documents(st.session_state.docs, hf_emb)
      print('Embedding done')
      print('Embedding done in', (time.time() - embedding_start) / 60, 'minutes')
      
      st.write(f'It took {(time.time() - start) / 60} minutes to load the data')

def create_retriever(prompt_template):
  """
  Creates a retriever chain using the ChatGroq model and the provided prompt template.

  Args:
      prompt_template (ChatPromptTemplate): The prompt template to be used in the retriever chain.

  Returns:
      RetrievalChain: The retriever chain with the ChatGroq model and the provided prompt template.
  """
  chat = ChatGroq(
    temperature=0.8,
    model="llama3-70b-8192",
    verbose=True
) 
  chain = create_stuff_documents_chain(
  prompt=prompt_template,
  llm=chat
  )
  retriever = st.session_state.vector.as_retriever()
  retriever_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=chain
  )
  return retriever_chain
def ask(retriever_chain, input):  
  """
  Invokes the retriever chain with the given input and writes the answer to the Streamlit output.

  Args:
      retriever_chain (RetrievalChain): The retriever chain to invoke.
      input (str): The input to the retriever chain.

  Returns:
      None
  """
  response = retriever_chain.invoke({"input": input})
  st.write(response["answer"])


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

query = st.text_input('Enter arxiv query number of the research paper (eg. 2312.00518)').strip()
      
if query:
  st.write('Please wait while we load the data...')
  load_vector(query)
  retriever_chain = create_retriever(prompt_template)
  st.write('Data loaded successfully!')
  input = st.text_input("Enter Your question about your specific research papers")
  ask(retriever_chain, input)






