from abc import ABC, abstractmethod

class AbstractLoader(ABC):
    @abstractmethod
    def create_loader(self, *args, **kwargs):
        """Subclasses must implement this to return the specific loader."""
        pass

    def load(self, *args, **kwargs):
        """Template method for loading and processing content."""
        # Pass **kwargs to the create_loader method
        loader = self.create_loader(*args, **kwargs)
        docs = loader.load()

        # Shared logic: concatenate content from all pages
        doc_text = ""
        for doc in docs:
            doc_text += "\n ... \n"
            doc_text += doc.page_content
        return doc_text
