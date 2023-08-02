import os
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load the OpenAI API key from the environment variable
load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

file_path = 'data/Hiker_Food.csv'
loader = CSVLoader(file_path=file_path)
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

# Create the chatbot
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(streaming=True),
    chain_type="stuff", # Modify as needed
    retriever=docsearch.vectorstore.as_retriever(),
    input_key="question",
    return_source_documents=True
)

# Function to summarize the response
def summarize_response(response):
    result = response['result']
    source_docs = response['source_documents']

    summary = f"Result: {result}\n\n"

    with st.expander("See source documents"):
        for doc in source_docs:
            page_content = doc.page_content.split('\n')
            brand_line = [line for line in page_content if "Brand:" in line][0]
            bold_brand_line = f"**{brand_line}**"  # Markdown syntax for bold
            other_lines = [line for line in page_content if "Brand:" not in line]
            table = bold_brand_line + " | " + " | ".join(other_lines)
            st.markdown(" - " + table + "\n\n")

    return summary

# Function to generate the chatbot's response
def conversational_chat(query):
    result = chain({"question": query})
    summary = summarize_response(result)
    return summary

# Accept user input
user_input = st.text_input("Query:")
if user_input:
    output = conversational_chat(user_input)
    st.write(output)
