import chainlit as cl
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
import os

load_dotenv()

# Load the env open ai key using uaer_api_key
user_api_key = os.getenv("OPENAI_API_KEY")

# Load your PDF data and initialize Langchain components
persist_directory = "./storage"
pdf_path = "./Politics_in_Zimbabwe.pdf"

# Use PyMuPDFLoader to load the document into the vector
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Split document content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# Load the data into Chroma db
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectordb.persist()

# Use the as_retriever to query the chuncked data inthe chroma db
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name='gpt-3.5-turbo')

# create a chain here
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Chainlit app functions

@cl.on_chat_start
def main():

    # Store the chain in the user session
    cl.user_session.set("qa", qa)


@cl.on_message
async def main(message: str):
    # Call the Langchain app to process the user's query
    try:
        llm_response = qa(f"###Prompt {message}")
        response = llm_response["result"]
    except Exception as err:
        response = 'Exception occurred. Please try again' + str(err)

    # Send the response
    await cl.Message(content=response).send()
