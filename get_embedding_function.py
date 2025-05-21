#from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.llms import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    """
    Returns a local embedding model served by Ollama.
    Make sure `ollama serve` is running and the model
    `nomic-embed-text` has been pulled.
    """
    return OllamaEmbeddings(model="nomic-embed-text")
