from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import os, re
from langchain_postgres.vectorstores import PGVector
from langchain.indexes import SQLRecordManager, index

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
        namespace = "pgvector/exobrain"
        self.record_manager = SQLRecordManager(namespace, db_url=self.vector_connection_string)
        self.record_manager.create_schema()

    def connection_string(self):
        return self.vector_connection_string
