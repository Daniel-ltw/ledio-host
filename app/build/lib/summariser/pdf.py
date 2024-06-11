from langchain.document_loaders import PyPDFLoader
from summariser.text import summarise_text

def summarise_pdf(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    text = ""
    for page in pages:
        text += page.page_content + '\n\n'
    return summarise_text(text)

