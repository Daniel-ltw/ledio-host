from .tools.html import load_html
from .tools.pdf import load_pdf

def save_html(filepath):
    loader = load_html(filepath)
    _save_embedding(loader)

def save_pdf(filepath):
    loader = load_pdf(filepath)
    _save_embedding(loader)

def _save_embedding(loader):
    from .tools.library import Library
    library = Library(loader)
    library.chunk()
    library.index()
