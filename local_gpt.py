from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata


class LocalGPT:
    model = None
    chain = None
    prompt = None

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def format_documents(self, documents):
        return "\n\n".join(document.page_content for document in documents)
    
    def load_and_split_documents(self, file_path: str):
        documents = PyPDFLoader(file_path=file_path).load()

        text_splitter = CharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        
        chunks = text_splitter.split_documents(documents)
        chunks = filter_complex_metadata(chunks)

        return chunks

    def create_and_store_embeddings(self, chunks):
        embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model, 
            collection_name="rag-chroma"
        )
        
        return vector_store

    def make_chain(self, file_path: str):
        chunks = self.load_and_split_documents(file_path)

        vector_store = self.create_and_store_embeddings(chunks)

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = (
            {"context": retriever | self.format_documents, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser()
        )

    def get_response(self, question: str):
        if not self.chain:
            return "Please upload a document."

        return self.chain.invoke(question)

    def clear(self):
        self.chain = None
