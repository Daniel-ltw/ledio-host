from langchain_community.document_loaders import PyPDFLoader

def load_pdf(filepath):
    return PyPDFLoader(filepath)
