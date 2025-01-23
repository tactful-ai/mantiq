from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

class Markdown:
    
    def __init__(self):
        pass

    def load(self,endpoint, key, file_path):
        
    
        loader = AzureAIDocumentIntelligenceLoader(
            
                api_endpoint=endpoint, api_key=key, file_path=file_path, api_model="prebuilt-layout"

         )
            
        docs = loader.load()

        # Concatenate the content of all pages
        doc_text = ""
        for doc in docs:
            doc_text += "\n ... \n"
            doc_text += doc.page_content
        
        return doc_text
