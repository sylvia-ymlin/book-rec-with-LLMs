from typing import List, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from src.config import DESCRIPTIONS_TXT, CHROMA_DB_DIR, EMBEDDING_MODEL
from src.utils import setup_logger

logger = setup_logger(__name__)


class VectorDB:
    """
    Singleton wrapper for the ChromaDB vector database.
    Uses local sentence-transformers for embedding generation.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDB, cls).__new__(cls)
            cls._instance.db = None
            cls._instance.embeddings = None
        return cls._instance

    def __init__(self):
        if self.db is None:
            self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB with local embeddings."""
        try:
            # Use local sentence-transformers model (no API calls)
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully")

            if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
                logger.info(f"Loading existing vector database from {CHROMA_DB_DIR}")
                self.db = Chroma(
                    persist_directory=str(CHROMA_DB_DIR),
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded {self.db._collection.count()} documents from vector database")
            else:
                error_msg = (
                    f"Vector Database not found at {CHROMA_DB_DIR}.\n"
                    "Please run the initialization script first to build the index:\n"
                    "    python src/init_db.py"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
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
