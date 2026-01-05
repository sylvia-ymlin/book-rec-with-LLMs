from typing import List, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from src.config import DESCRIPTIONS_TXT, CHROMA_DB_DIR, EMBEDDING_MODEL, HF_TOKEN
from src.utils import setup_logger

logger = setup_logger(__name__)

class VectorDB:
    """
    Singleton wrapper for the ChromaDB vector database.
    Ensures only one instance of the database connection exists.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDB, cls).__new__(cls)
            cls._instance.db = None
        return cls._instance

    def __init__(self):
        if self.db is None:
            self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB with persistence."""
        try:
            embeddings = HuggingFaceEndpointEmbeddings(
                model=EMBEDDING_MODEL,
                huggingfacehub_api_token=HF_TOKEN
            )

            if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
                logger.info(f"Loading existing vector database from {CHROMA_DB_DIR}")
                self.db = Chroma(
                    persist_directory=str(CHROMA_DB_DIR),
                    embedding_function=embeddings
                )
            else:
                logger.info("Creating new vector database...")
                raw_documents = TextLoader(str(DESCRIPTIONS_TXT)).load()
                text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
                documents = text_splitter.split_documents(raw_documents)
                
                self.db = Chroma.from_documents(
                    documents,
                    embedding=embeddings,
                    persist_directory=str(CHROMA_DB_DIR)
                )
                logger.info(f"Vector database created and saved to {CHROMA_DB_DIR}")
                
        except Exception as e:
            logger.error(f"Error initializing Vector DB: {str(e)}")
            raise

    def search(self, query: str, k: int = 50) -> List[Any]:
        """
        Perform a semantic similarity search on the vector database.
        
        Args:
            query (str): The user's natural language query.
            k (int): Number of results to retrieve.
            
        Returns:
            List[Document]: List of LangChain Document objects matching the query.
        """
        return self.db.similarity_search(query, k=k)
