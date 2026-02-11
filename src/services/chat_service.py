from typing import Generator, Optional, Dict, Any, List
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

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
    _history: Dict[str, List[BaseMessage]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Data is now loaded lazily via _ensure_data
        pass

    def _ensure_data(self):
        if self._books_df is None:
            logger.info("ChatService: Lazy-loading books data for context retrieval...")
            self._books_df = load_books_data()

    def _get_book_context(self, isbn: str) -> Optional[Dict[str, Any]]:
        """Retrieve full context for a specific book by ISBN."""
        self._ensure_data()
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
        
        # Add Review Highlights
        reviews = book.get('review_highlights', '')
        if reviews:
             info += f"Review Highlights (What readers say): {reviews}\n"
             
        return info

    def _get_history_key(self, user_id: str, isbn: str) -> str:
        return f"{user_id}:{isbn}"

    def _update_history(self, key: str, human: str, ai: str):
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(HumanMessage(content=human))
        self._history[key].append(AIMessage(content=ai))
        # Limit to last 10 messages (5 turns)
        if len(self._history[key]) > 10:
            self._history[key] = self._history[key][-10:]

    def clear_history(self, user_id: str, isbn: str):
        key = self._get_history_key(user_id, isbn)
        if key in self._history:
            del self._history[key]

    async def chat_stream(
        self, 
        isbn: str, 
        user_query: str, 
        user_id: str = "local",
        api_key: Optional[str] = None,
        provider: str = "ollama"
    ) -> Generator[str, None, None]:
        """
        Stream chat response for a specific book.
        """
        self._ensure_data()
        # 1. Fetch Context
        book = self._get_book_context(isbn)
        if not book:
            yield "I'm sorry, I couldn't find the details for this book."
            return

        # 2. Build Persona (User Profile)
        favs = list_favorites(user_id)
        persona_data = build_persona(favs, self._books_df)
        user_persona = persona_data.get("summary", "General Reader")

        # 3. Construct Prompt with History
        from src.core.context_compressor import compressor
        
        # Compress History (if needed)
        key = self._get_history_key(user_id, isbn)
        raw_history = self._history.get(key, [])
        compressed_history = []
        
        # We need to await the async function, but we are in an async generator.
        # This is fine.
        try:
            compressed_history = await compressor.compress_history(raw_history)
        except:
             compressed_history = raw_history[-10:] # Fallback

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
            *compressed_history,
            HumanMessage(content=user_query)
        ]

        # 4. Invoke LLM (Streaming)
        full_response = ""
        try:
            llm = LLMFactory.create(provider=provider, api_key=api_key, temperature=0.5)
            # Use astream for async streaming
            async for chunk in llm.astream(messages):
                content = chunk.content
                full_response += content
                yield content
            
            # 5. Save functionality
            self._update_history(key, user_query, full_response)
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            yield f"Error generating response: {str(e)}. Please check your API Key."

    def add_book_to_context(self, book_data: Dict[str, Any]):
        """Dynamically add a new book to the ChatService context."""
        self._ensure_data()
        try:
            if self._books_df is not None:
                new_row_df = pd.DataFrame([book_data])
                self._books_df = pd.concat([self._books_df, new_row_df], ignore_index=True)
                logger.info(f"ChatService: Added book {book_data.get('isbn13')} to context.")
        except Exception as e:
            logger.error(f"ChatService: Failed to add book to context: {e}")

def get_chat_service():
    """Helper for lazy access to the ChatService singleton."""
    return ChatService()

chat_service = ChatService()
