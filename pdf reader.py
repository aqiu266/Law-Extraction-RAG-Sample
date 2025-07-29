import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Loads the PDF
pdf_path = "C:/Users/User/Documents/0.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Splits the document up into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Builds RetrievalQA chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# I/O
while True:
    query = input("\nAsk a question about the PDF (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain({"query": query})
    print("\nAnswer:\n", result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"), "|", doc.page_content[:200], "...\n")
