from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import os, re
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_postgres.vectorstores import PGVector
from langchain.indexes import SQLRecordManager, index
from langchain.docstore.document import Document

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate

class Library:
    def __init__(self, loader=None):
        if loader is None:
            raise "Loader not provided"

        self.loader = loader
        store = Store()
        self.embedding = Embedding()
        self.embedding_function = self.embedding.function()
        self.vector_connection_string = store.connection_string()
        self.record_manager = store.record_manager

    def chunk(self):
        semantic_chunker = SemanticChunker(embeddings=self.embedding_function, breakpoint_threshold_type='standard_deviation', sentence_split_regex="\\n|\\xa0|\\s\\s")
        semantic_chunks = semantic_chunker.split_documents(self.loader.load())
        print(len(semantic_chunks))

        for chunk in semantic_chunks:
            chunk.metadata['tags'] = re.split("\.|-|\/", chunk.metadata['source'])

        tokenizer = AutoTokenizer.from_pretrained(self.embedding.embedding_model)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=20,length_function=lambda text: len(tokenizer.tokenize(text, truncation=True)),is_separator_regex=False)
        self.chunks = text_splitter.split_documents(semantic_chunks)
        print(len(self.chunks))

    def index(self):
        vectorstore = PGVector(
            collection_name="ledio_host",
            connection=self.vector_connection_string,
            embeddings = self.embedding_function,
        )

        # 9 index documents to prevent being readded
        result = index(self.chunks,
            self.record_manager,
            vectorstore,
            cleanup=None,
            source_id_key='tags',
        )
        print(result)


class Embedding:
    def __init__(self):
        self.embedding_model = "BAAI/bge-small-en-v1.5"
        self.embedding_model_kwargs = {'device': 'cpu'}
        self.embedding_encode_kwargs = {'normalize_embeddings': True}
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model,
            model_kwargs=self.embedding_model_kwargs,
            encode_kwargs=self.embedding_encode_kwargs,
        )

    def function(self):
        return self.embedding



class Retrievers:
    def __init__(self, model=None, handler=None, pg_vector_store=None, record_manager=None):
        self.duckduckgo = DuckDuckGoSearchRun()
        self.wikipedia = WikipediaRetriever()
        self.handler = handler
        self.model = model
        self.embedding = Embedding()
        self.embedding_function = self.embedding.function()
        self.pg_vector_store = pg_vector_store
        self.record_manager = record_manager

    def search(self, query):
        if self._exist_in_db(query):
            return self

        self.wiki_docs = self.wikipedia.invoke(query)[0]
        self.ddg_docs = Document(page_content=self.duckduckgo.run(query), metadata={"source": f"https://duckduckgo.com/?q={query.replace(' ', '+')}"} )

        if self.pg_vector_store is not None:
            summary = Document(page_content=self._summary(), metadata={"source": f"summary: {query}"} )
            self._chunk(summary, query)
            self._index()

        return self

    def verify(self, user_input):
        template = """
            You are a helpful assistant.

            Please verify if this song exist: {song}

            Summary: {summary}

            Answer in 'yes' and 'no'.
        """
        prompt = ChatPromptTemplate.from_template(template)

        compressor = LLMChainExtractor.from_llm(self.model)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.pg_vector_store.as_retriever(search_kwargs={"body_search": user_input})
        )

        chain = ( { 'summary': compression_retriever, 'song': RunnablePassthrough() } | prompt | self.model )

        result = chain.invoke(user_input, {"callbacks": [self.handler]})
        # print(user_input, ': ', result.lower())
        return 'yes' in result.lower()

    def _chunk(self, doc, query):
        semantic_chunker = SemanticChunker(embeddings=self.embedding_function, breakpoint_threshold_type='standard_deviation', sentence_split_regex="\\n|\\xa0|\\s\\s")
        semantic_chunks = semantic_chunker.split_documents([doc])
        # print(len(semantic_chunks))

        for chunk in semantic_chunks:
            chunk.metadata['tags'] = list(filter(None, re.split("\.|-|\/|\(|\)|\_|:| ", query)))

        tokenizer = AutoTokenizer.from_pretrained(self.embedding.embedding_model)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=20,length_function=lambda text: len(tokenizer.tokenize(text, truncation=True)),is_separator_regex=False)
        self.chunks = text_splitter.split_documents(semantic_chunks)
        # print(len(self.chunks))


    def _index(self):
        index(self.chunks,
            self.record_manager,
            self.pg_vector_store,
            cleanup=None,
            source_id_key='tags',
        )
        # print(result)

    def _summary(self):
        template = f"""
            You are a helpful assistant.

            Can you give me a summary of the following:

            wikipedia result: {self.wiki_docs.page_content}
            duckduckgo result: {self.ddg_docs.page_content}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = ( prompt | self.model )

        return chain.invoke({"callbacks": [self.handler]})

    def _exist_in_db(self, query):
        template = """
            You are a helpful assistant.

            Please verify if the following is in the vectorstore: {question}

            vectorstore: {context}

            Answer in 'yes' and 'no'.
        """
        prompt = ChatPromptTemplate.from_template(template)

        compressor = LLMChainExtractor.from_llm(self.model)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.pg_vector_store.as_retriever(search_kwargs={"body_search": query})
        )

        chain = (
            {"context": compression_retriever, "question": RunnablePassthrough()}
            | prompt | self.model
        )

        result = chain.invoke(query, {"callbacks": [self.handler]})
        return 'yes' in result.lower()




class Store:
    def __init__(self):
        self.vector_connection_string = PGVector.connection_string_from_db_params(
            database=os.getenv('POSTGRES_DB', 'postgres'),
            driver='psycopg',
            host=os.getenv('POSTGRES_HOST', 'db'),
            port=5432,
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
        )
        self.db_connection_string = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('POSTGRES_HOST', 'db')}:5432/{os.getenv('POSTGRES_DB', 'postgres')}"
        namespace = "pgvector/ledio"
        self.record_manager = SQLRecordManager(namespace, db_url=self.vector_connection_string)
        self.record_manager.create_schema()

    def connection_string(self):
        return self.vector_connection_string

    def vector_store(self):
        return PGVector(
            collection_name="ledio",
            connection=self.connection_string(),
            embeddings=Embedding().function(),
        )
