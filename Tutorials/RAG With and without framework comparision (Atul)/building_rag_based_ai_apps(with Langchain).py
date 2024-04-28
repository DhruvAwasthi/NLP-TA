from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
loader = CSVLoader(file_path='./naval.csv', csv_args={
    'delimiter': ',',
    'fieldnames': ['SNO', 'Chapter Title', 'Chapter URL','Chapter Content']
}, source_column="Chapter URL")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key='OPENAI_API_KEY')
vectorstore = Chroma.from_documents(texts, embeddings)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_with_sources_chain(llm, chain_type="stuff")
query = "How to earn money?"
docs = vectorstore.similarity_search(query)
chain.run(input_documents=docs, question=query)