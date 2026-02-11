from typing import List, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from src.core.llm import LLMFactory
from src.utils import setup_logger

logger = setup_logger(__name__)

class ContextCompressor:
    """
    Service to compress RAG context and Conversation History.
    Reduces token usage and 'Lost in the Middle' phenomenon.
    """
    
    def __init__(self):
        # We use a cheaper/faster model for summarization if possible
        # For now, we reuse the default provider from LLMFactory
        pass

    async def compress_history(self, history: List[BaseMessage], max_token_limit: int = 2000) -> List[BaseMessage]:
        """
        Compress conversation history if it exceeds limits.
        Strategy: Keep last N messages raw, summarize the rest.
        """
        # Simple heuristic: If history > 10 messages, summarize the oldest ones
        if len(history) <= 6:
            return history
            
        # Keep last 4 messages (2 turns) intact
        recent_history = history[-4:]
        older_history = history[:-4]
        
        # If older history is small, just return (avoid unnecessary summarization calls)
        if len(older_history) < 2:
            return history

        logger.info(f"Compressing history: {len(history)} messages -> Summary + 4 recent")
        
        try:
            summary = await self._summarize_messages(older_history)
            return [SystemMessage(content=f"Previous Conversation Summary: {summary}")] + recent_history
        except Exception as e:
            logger.error(f"History compression failed: {e}")
            return history # Fallback: return full history (or could slice)

    async def _summarize_messages(self, messages: List[BaseMessage]) -> str:
        """Use LLM to summarize a list of messages."""
        conversation_text = ""
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            conversation_text += f"{role}: {msg.content}\n"
            
        prompt = (
            "Summarize the following conversation concisely, focusing on key user preferences and questions. "
            "Do not lose important details.\n\n"
            f"{conversation_text}"
        )
        
        # Use simple mock if running in test environment/benchmark without keys
        try:
            llm = LLMFactory.create(temperature=0.3)
        except:
             # Fallback to mock for stability if env is not set
             llm = LLMFactory.create(provider="mock")

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def format_docs(self, docs: List[Any], max_len_per_doc: int = 500) -> str:
        """
        Format retrieved documents for the LLM Prompt.
        Truncates content to avoid context overflow.
        """
        formatted = ""
        for i, doc in enumerate(docs):
            content = doc.page_content.replace("\n", " ")
            if len(content) > max_len_per_doc:
                content = content[:max_len_per_doc] + "..."
            
            # Add Relevance Score if available (from Reranker)
            score_info = ""
            if doc.metadata and "relevance_score" in doc.metadata:
                score = doc.metadata["relevance_score"]
                score_info = f" (Relevance: {score:.2f})"
                
            formatted += f"[{i+1}] {content}{score_info}\n"
        return formatted

# Singleton
compressor = ContextCompressor()
