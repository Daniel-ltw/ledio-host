from langchain_community.document_loaders import BSHTMLLoader

def load_html(filepath):
    return BSHTMLLoader(filepath)
