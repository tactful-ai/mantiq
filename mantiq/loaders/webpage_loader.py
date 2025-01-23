from langchain_community.document_loaders import WebBaseLoader
from .abstract_loader import AbstractLoader

class WebPage(AbstractLoader):
    def create_loader(self, web_paths, **kwargs):
        # Pass **kwargs to WebBaseLoader
        return WebBaseLoader(web_paths=web_paths, **kwargs)
