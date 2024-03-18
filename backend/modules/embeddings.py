from langchain.embeddings import HuggingFaceEmbeddings
import os


class EmbeddingsInstance:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("DEBUG", "Embeddings Model initialized")
            cls.embedding_model = HuggingFaceEmbeddings(model_name=os.getenv('EMBEDDINGS_MODEL_PATH'))
        return cls._instance
