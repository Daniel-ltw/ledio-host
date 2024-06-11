import os
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from langfuse.callback import CallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from langchain_postgres.vectorstores import PGVector

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# from .tools.library import Embedding, Store

handler = CallbackHandler(
        os.getenv('LANGFUSE_PUBLIC_KEY'),
        os.getenv('LANGFUSE_SECRET_KEY'),
        host="http://langfuse:3000",
        )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", token=os.getenv("HUGGINGFACE_TOKEN"))
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HUGGINGFACE_TOKEN"))



# vectorstore = PGVector(
#     collection_name="ledio_host",
#     connection=Store().connection_string(),
#     embeddings =Embedding().function(),
# )

def chat():
    try:
        while True:
            print("\n> ", end="")
            user_input = input()
            if user_input in ['/exit', 'exit']:
                bye()
                break

            prompt = basic_chatters()
            chain = (prompt | model)

            # if in_vector(user_input):
            #     prompt = query_with_context()

            #     compressor = LLMChainExtractor.from_llm(model)
            #     compression_retriever = ContextualCompressionRetriever(
            #         base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_kwargs={"body_search": user_input})
            #     )

            #     chain = (
            #         {"context": compression_retriever, "question": RunnablePassthrough()}
            #         | prompt | model
            #     )

            for s in chain.stream({'input': user_input}, {"callbacks": [handler]}):
                print(s.content, end="", flush=True)
    except EOFError:
       bye()

def bye():
    print('so long, farewell')

def basic_chatters():
    template = "You are a helpful assistant. {input}"
    return ChatPromptTemplate.from_template(template)

def query_with_context():
    template = """Answer the question based only on the following context: {context}
        ----
        Question: {question}
    """
    return ChatPromptTemplate.from_template(template)

def in_vector(input):
    template = """
        You are a helpful assistant.

        Please verify if the following is in the vectorstore: {question}

        vectorstore: {context}

        Answer in 'yes' and 'no'.
    """
    prompt = ChatPromptTemplate.from_template(template)

    compressor = LLMChainExtractor.from_llm(model)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_kwargs={"body_search": input})
    )

    chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | prompt | model
    )

    result = chain.invoke(input, {"callbacks": [handler]})
    return result.content == 'yes'
