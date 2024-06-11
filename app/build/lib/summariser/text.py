from langchain.chat_models import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter

# gets COHERE_API_KEY from environment automatically
model = ChatCohere(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

def summarise_file(filename):
    with open(filename, "r") as f:
        text = f.read()
        summarise_text(text)

def summarise_text(text):
    template = """Please create a short summary of this: {text}"""    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = prompt | model

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.create_documents([text])

    summaries = ""
    if len(texts) > 1:
        print(f"Sending {len(texts)} texts to LLM")
    for text in texts:
        summary = chain.invoke({"text": text}).content
        summaries = summaries + "\n----\n" + summary
        if (len(texts) > 1):
            print('\n----')

    if (len(texts) > 1):
        print('***final summary***')
        chain.invoke({"text": summaries})
