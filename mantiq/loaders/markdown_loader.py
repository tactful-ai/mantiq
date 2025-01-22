from langchain_community.document_loaders import UnstructuredMarkdownLoader
from .abstract_loader import AbstractLoader

class Markdown(AbstractLoader):
    def create_loader(self, file_path, **kwargs):
        # Pass **kwargs to UnstructuredMarkdownLoader
        return UnstructuredMarkdownLoader(file_path, **kwargs)
