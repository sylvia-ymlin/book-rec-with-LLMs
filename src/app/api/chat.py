from fastapi import APIRouter, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from src.services.chat_service import chat_service
from src.infra.utils import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    """Request for streaming chat about a book."""

    isbn: str = Field(..., description="ISBN of the book to chat about")
    query: str = Field(..., description="User question (e.g. 'What is the main theme?')")
    user_id: Optional[str] = Field(default="local", description="User identifier")
    provider: Optional[str] = Field(
        default="ollama",
        description="LLM provider: 'ollama' | 'openai' | 'groq' | 'deepseek'",
    )

    model_config = {"json_schema_extra": {"examples": [{"isbn": "0140283331", "query": "What is the main theme of this book?"}]}}


async def get_llm_key(x_llm_key: Optional[str] = Header(None, alias="X-LLM-Key")):
    """Dependency to extract API Key from header."""
    return x_llm_key


@router.post(
    "/completions",
    summary="Stream chat",
    description="""Stream chat response for a book using RAG + LLM.
- **RAG**: Retrieves relevant passages from the book, injects into context
- **Streaming**: Returns `text/plain` chunks (Server-Sent Events style)
- **Headers**: `X-LLM-Key` required for cloud providers (`openai` / `groq` / `deepseek`)
- **Provider**: `ollama` (default, local) | `openai` | `groq` | `deepseek`
""",
    responses={
        200: {"description": "Stream of text chunks"},
        500: {"description": "LLM or RAG error"},
    },
)
async def chat_completions(
    request: ChatRequest,
    api_key: Optional[str] = Depends(get_llm_key),
):
    """Stream chat response for a book using RAG + LLM."""
    logger.info(f"Chat request: isbn={request.isbn}, query='{request.query}', provider={request.provider}")

    # Check if provider is openai and key is missing
    if request.provider == "openai" and not api_key:
        # LLMFactory checks env var too. So we pass None and let it fail if needed.
        pass

    return StreamingResponse(
        chat_service.chat_stream(
            isbn=request.isbn,
            user_query=request.query,
            user_id=request.user_id,
            api_key=api_key,
            provider=request.provider,
        ),
        media_type="text/plain",
    )

