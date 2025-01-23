from langchain_community.document_loaders import UnstructuredHTMLLoader
from .abstract_loader import AbstractLoader

class HTML(AbstractLoader):
    def create_loader(self, file_path, **kwargs):
        # Pass **kwargs to UnstructuredHTMLLoader
        return UnstructuredHTMLLoader(file_path, **kwargs)
