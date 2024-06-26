import os
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langfuse.callback import CallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.llms import Ollama
from transformers import AutoTokenizer
from uuid import uuid4
from datetime import datetime
from .tools.library import Store, Retrievers



store = Store()

def host():
    session_id = str(uuid4())
    history = PostgresChatMessageHistory(
        connection_string=store.db_connection_string,
        session_id=session_id,
    )
    handler = CallbackHandler(
        os.getenv('LANGFUSE_PUBLIC_KEY'),
        os.getenv('LANGFUSE_SECRET_KEY'),
        host="http://langfuse:3000",
        session_id=session_id,
    )
    index = 0
    model = Ollama(base_url="http://192.168.172.64:11434", callbacks=[handler], model="wangshenzhi/llama3-8b-chinese-chat-ollama-q8")
    # model = ChatGroq(callbacks=[handler], model="llama3-70b-8192")
    pg_vector_store = store.vector_store()
    record_manager = store.record_manager
    retrievers = Retrievers(model, handler, pg_vector_store, record_manager)

    try:
        while True:
            print("\n> ", end="")
            user_input = input()
            if user_input in ['/exit', 'exit']:
                bye()
                break

            result = retrievers.search(user_input).verify(user_input)

            if not result:
                history.add_user_message(user_input)
                response = 'The song does not exist. Please try again.'
                print(response, end="", flush=True)
                history.add_ai_message(response)
                continue

            prompt = host_script(history)

            compressor = LLMChainExtractor.from_llm(model)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=pg_vector_store.as_retriever(search_kwargs={"body_search": user_input})
            )

            chain = ( { 'summary': compression_retriever, 'input': RunnablePassthrough() } | prompt | model )

            response = ''
            for stream in chain.stream(user_input, { "callbacks": [handler] }):
                print(stream, end="", flush=True)
                response = response + stream

            history.add_user_message(user_input)
            history.add_ai_message(response)
            produce_audio(f"{index:02d} - {user_input}", response)
            summarise_session(model, handler, history)
            index = index + 1
    except EOFError:
       bye()

def bye():
    print('so long, farewell')

def host_script(history):
    template = ''
    now = datetime.today()
    if len(history.messages) > 0:
        template = template + f"Date and time: {now.strftime('%d %B %Y - %H:%M:%S')}"
    else:
        template = template + f"You need to introduce the day with date and time: {now.strftime('%d %B %Y - %H:%M:%S')}"

    template = template + """
        You are a experience and expressive music host named Jay.

        When given a song metadata and artist metadata, please produce a short script to introduce the song and artist.

        ----

        Song - {input}

        Summary: {summary}
    """

    human = HumanMessagePromptTemplate.from_template(template)

    return (ChatPromptTemplate.from_messages(history.messages) + human)

def summarise_session(model, handler, history):
    model_id = "meta-llama/Meta-Llama-3-70b-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HUGGINGFACE_TOKEN"))
    token_set = ''
    for message in history.messages:
        token_set += message.type + ": " + message.content + "\n"
    tokens = tokenizer.encode(token_set)
    print('token count:', len(tokens))

    if len(tokens) > 1000:
        template = """
            You are an expert at summarising conversations.

            Please summarise the following conversation?

            {input}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = ( prompt | model )

        result = chain.invoke(token_set, {"callbacks": [handler]})
        history.clear()
        summary = SystemMessage(content=result)
        history.add_message(summary)

def produce_audio(name, script):
    import torch
    from TTS.api import TTS

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

    # Run TTS
    # ❗ Since this model is multi-lingual model, we must set the target speaker and language
    # Text to speech to a file
    tts.tts_to_file(text=script, language='en', speaker='Aaron Dreschner', file_path=f"{name}.wav".lower())


