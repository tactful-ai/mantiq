from langchain_community.document_loaders import CSVLoader
from .abstract_loader import AbstractLoader

class CSV(AbstractLoader):
    def create_loader(self, file_path, **kwargs):
        # Pass **kwargs to CSVLoader to allow additional options
        return CSVLoader(file_path, **kwargs)
