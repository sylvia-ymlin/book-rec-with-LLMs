from typing import Generator, Optional, Dict, Any
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm import LLMFactory
from src.etl import load_books_data
from src.marketing.persona import build_persona
from src.user.profile_store import list_favorites
from src.utils import setup_logger

logger = setup_logger(__name__)

class ChatService:
    """
    Service for RAG-based chat interaction.
    Currently focused on 'Chat with Book' (Single Item Context).
    """
    _instance = None
    _books_df = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._books_df is None:
            logger.info("ChatService: Loading books data for context retrieval...")
            self._books_df = load_books_data()

    def _get_book_context(self, isbn: str) -> Optional[Dict[str, Any]]:
        """Retrieve full context for a specific book by ISBN."""
        # Handle string/int types for ISBN
        try:
            row = self._books_df[self._books_df["isbn13"].astype(str) == str(isbn)]
            if row.empty:
                return None
            return row.iloc[0].to_dict()
        except Exception:
            return None

    def _format_book_info(self, book: Dict[str, Any]) -> str:
        """Format book metadata into a readable context string."""
        info = f"Title: {book.get('title', 'Unknown')}\n"
        info += f"Author: {book.get('authors', 'Unknown')}\n"
        info += f"Category: {book.get('simple_categories', 'General')}\n"
        info += f"Tags: {book.get('tags', '')}\n"
        
        # Emotions
        emotions = []
        for e in ['joy', 'sadness', 'fear', 'anger', 'surprise']:
            if float(book.get(e, 0) or 0) > 0.3:
                emotions.append(e)
        if emotions:
            info += f"Emotional Tone: {', '.join(emotions)}\n"
            
        info += f"Description: {book.get('description', 'No description available.')}\n"
        return info

    async def chat_stream(
        self, 
        isbn: str, 
        user_query: str, 
        user_id: str = "local",
        api_key: Optional[str] = None,
        provider: str = "openai"
    ) -> Generator[str, None, None]:
        """
        Stream chat response for a specific book.
        """
        # 1. Fetch Context
        book = self._get_book_context(isbn)
        if not book:
            yield "I'm sorry, I couldn't find the details for this book."
            return

        # 2. Build Persona (User Profile)
        favs = list_favorites(user_id)
        persona_data = build_persona(favs, self._books_df)
        user_persona = persona_data.get("summary", "General Reader")

        # 3. Construct Prompt
        system_prompt = (
            "You are a knowledgeable and helpful intelligent librarian. "
            "The user is asking questions about a specific book. "
            "Use the provided book context to answer accurately. "
            "If the answer is not in the context, use your general knowledge but mention that it's outside the provided text. "
            "Keep answers concise and conversational.\n\n"
            f"--- BOOK CONTEXT ---\n{self._format_book_info(book)}\n"
            f"--- USER PERSONA ---\n{user_persona}\n"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]

        # 4. Invoke LLM (Streaming)
        try:
            llm = LLMFactory.create(provider=provider, api_key=api_key, temperature=0.5)
            # Use astream for async streaming
            async for chunk in llm.astream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            yield f"Error generating response: {str(e)}. Please check your API Key."

chat_service = ChatService()
