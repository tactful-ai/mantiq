from langchain_community.document_loaders import JSONLoader
from .abstract_loader import AbstractLoader

class JSON(AbstractLoader):
    def __init__(self, jq_schema=".data[].attributes.message"):
        self.jq_schema = jq_schema

    def create_loader(self, file_path, **kwargs):
        # Pass **kwargs along with default jq_schema
        return JSONLoader(file_path, jq_schema=self.jq_schema, text_content=False, **kwargs)
