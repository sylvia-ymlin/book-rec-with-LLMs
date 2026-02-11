from fastapi import APIRouter, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from src.services.chat_service import chat_service
from src.utils import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    isbn: str
    query: str
    user_id: Optional[str] = "local"
    provider: Optional[str] = "openai"  # openai, ollama

async def get_llm_key(x_llm_key: Optional[str] = Header(None, alias="X-LLM-Key")):
    """Dependency to extract API Key from header."""
    # For Ollama, key is optional. For OpenAI, it's required (enforced by LLMFactory).
    return x_llm_key

@router.post("/completions")
async def chat_completions(
    request: ChatRequest,
    api_key: Optional[str] = Depends(get_llm_key)
):
    """
    Stream chat response for a book using RAG + LLM.
    Requires 'X-LLM-Key' header for OpenAI.
    """
    logger.info(f"Chat request: isbn={request.isbn}, query='{request.query}', provider={request.provider}")
    
    # Check if provider is openai and key is missing
    if request.provider == "openai" and not api_key:
         # Check env var fallback inside service/factory, but good to warn here?
         # LLMFactory checks env var too. So we pass None and let it fail if needed.
         pass

    return StreamingResponse(
        chat_service.chat_stream(
            isbn=request.isbn, 
            user_query=request.query, 
            user_id=request.user_id,
            api_key=api_key,
            provider=request.provider
        ),
        media_type="text/plain"
    )
