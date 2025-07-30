#my code: replaces messages[i][0]?

case_text = ""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def preProcessCaseText(query, full_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200) # based on this article it seems like this is the optimal chunk split: https://research.trychroma.com/evaluating-chunking
    chunks = splitter.split_text(full_text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # chunk and vectorize full_text
    # get a few relevant chunks based on similarity to query



    chunks = ["", ""]
    context = ""
    for chunk in chunks:
        context += f"<case_text_chunk>\n{chunk}\n</case_text_chunk>"

    return context
    context = [f"<case_text_chunk>\n{chunk}\n</case_text_chunk>" for chunk in chunks] #turn it into a single text prompt / concatenate it

"""
llm = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
"""

# big problem with llms rn: can't guarantee outputs are correct
# can't verify reasoning in natural language
# hendrix is trying to extract a knowledge graph to try to build a space of everything that's going on
