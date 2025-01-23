from .csv_loader import CSV
from .markdown_loader import Markdown
from .json_loader import JSON
from .html_loader import HTML
from .directory_loader import Directory
from .webpage_loader import WebPage
from .pdf_loader import PDF

class LoaderFactory:
    @staticmethod
    def get_loader(loader_type, *args, **kwargs):
        loaders = {
            "CSV": CSV,
            "Markdown": Markdown,
            "JSON": JSON,
            "HTML": HTML,
            "Directory": Directory,
            "WebPage": WebPage,
            "PDF": PDF,
        }
        if loader_type in loaders:
            # Pass **kwargs to the loader's load method
            return loaders[loader_type]().load(*args, **kwargs)
        raise ValueError(f"Loader type '{loader_type}' is not supported")
