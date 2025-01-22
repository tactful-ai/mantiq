from langchain_community.document_loaders import DirectoryLoader
from .abstract_loader import AbstractLoader

class Directory(AbstractLoader):
    def create_loader(self, file_path, glob="*", **kwargs):
        # Pass **kwargs to DirectoryLoader
        return DirectoryLoader(file_path, glob=glob, **kwargs)
