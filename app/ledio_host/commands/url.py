from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

def save_url(url):
  loader = AsyncHtmlLoader(url)
  docs = loader.load()
  transformer = Html2TextTransformer()
  transformer.load(docs)
