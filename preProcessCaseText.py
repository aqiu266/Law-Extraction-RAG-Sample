#my code: replaces messages[i][0]?

case_text = ""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def preProcessCaseText(query, full_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000, separators = ["\n\n", "\n", " ", ".", "?", ""]) # based on this article it seems like this is the optimal chunk split: https://research.trychroma.com/evaluating-chunking - we assume that each token corresponds to a word and that the average word length is 5 characters
    chunks = splitter.split_text(full_text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # chunk and vectorize full_text
    # get a few relevant chunks based on similarity to query

    chunks = retriever._get_relevant_documents(query)
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
