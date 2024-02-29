from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
import time

if __name__ == '__main__':
    url = "https://www.kkbox.com/sg/en/song/8kFm4nAIsZ01P7mWgu"
    output_parser = StrOutputParser()
    # init ollama
    # llm = Ollama(model="tinyllama")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://89.116.167.82:11434")
    model = ChatOllama(model="tinyllama", base_url="http://89.116.167.82:11434")

    # load website data
    loader = WebBaseLoader(url)
    docs = loader.load()

    # prepare vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500)
    document = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(document, embeddings)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context.

        <context>
        {context}
        </context>
        
        Question: {input}
    """)

    # setup chain
    runnable = RunnablePassthrough
    retriever = vector.as_retriever()

    chain = (
        runnable.assign(context=(lambda x: x["input"]) | retriever)
        | prompt
        | model
        | output_parser
    )

    while True:
        start_time = time.time()
        question = input("\nAsk me:\n")
        if question == "stop":
            break
        for s in chain.stream({"input": question}):
            print(s, end="", flush=True)

        print("--- %s seconds ---" % (time.time() - start_time))

