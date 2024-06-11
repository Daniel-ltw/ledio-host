# 1 let's read a html doc
FILENAME='docs/eu-ai-act.html'
from langchain_community.document_loaders import BSHTMLLoader
loader = BSHTMLLoader(FILENAME)
#big_doc = loader.load()[0]

# 2 slice it up into small chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )
#chunks = loader.load_and_split(text_splitter)

# 3 let's be more accurate in slicing

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    length_function=lambda text: len(tokenizer.tokenize(text, truncation=True)),
    is_separator_regex=False,
)
# chunks = loader.load_and_split(text_splitter)
# print(len(chunks))

# 4 let's get the embedding for it
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': True}
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=EMBEDDING_MODEL_KWARGS,
    encode_kwargs=EMBEDDING_ENCODE_KWARGS,
)
#embedding = embedding_function.embed_query(chunks[0].page_content)

# 5 let's get the embeddings for every chunk
#all_embeddings = []
#for i, chunk in enumerate(chunks):
#    all_embeddings.append(embedding_function.embed_query(chunk.page_content))
#    print(f"chunk {i+1}/{len(chunks)} done")

# 6 connection string
import os
from langchain_postgres.vectorstores import PGVector

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    database=os.getenv('POSTGRES_DB', 'postgres'),
    driver='psycopg',
    host=os.getenv('POSTGRES_HOST', 'db'),
    port=5432,
    user=os.getenv('POSTGRES_USER', 'postgres'),
    password=os.getenv('POSTGRES_PASSWORD', 'password'),
)

# 7 save documents into the database
vectorstore = PGVector(
    collection_name="exobrain",
    connection=CONNECTION_STRING,
    embeddings = embedding_function,
)
#vectorstore.add_documents(chunks)
#res = vectorstore.similarity_search("penalties", k=3)
#print(res)

# 8 setup an index
# from langchain.indexes import SQLRecordManager
# namespace = "pgvector/exobrain"
# record_manager = SQLRecordManager(
#     namespace, db_url=CONNECTION_STRING
# )
# record_manager.create_schema()

# 9 index documents to prevent being readded
# from langchain.indexes import index
# result = index(chunks,
#     record_manager,
#     vectorstore,
#     cleanup=None,
#     source_id_key="source",
# )

# 10 augment a question
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough

#model = ChatGroq(model="llama3-70b-8192")
#template = """Answer the question based only on the following context: {context}
#----
#Question: {question}
#"""
#prompt = ChatPromptTemplate.from_template(template)
#chain = (
#        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
#        | prompt | model
#        )

#for s in chain.stream("what are the penalties for breaching the EU AI ACT"):
#    print(s.content, end="", flush=True)

# 11 integrate with langfuse
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host="http://langfuse:3000",
)
template = """Answer the question based only on the following context: {context}
----
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatGroq(model="llama3-8b-8192", callbacks=[langfuse_handler])
chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt | model
)

for s in chain.stream("how are you today?", {"callbacks": [langfuse_handler]}):
    print(s.content, end="", flush=True)
