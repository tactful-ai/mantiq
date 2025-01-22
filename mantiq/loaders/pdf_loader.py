from langchain_community.document_loaders import PyPDFLoader
from .abstract_loader import AbstractLoader

class PDF(AbstractLoader):
    def create_loader(self, file_path, **kwargs):
        # Pass **kwargs to PyPDFLoader
        return PyPDFLoader(file_path, **kwargs)
