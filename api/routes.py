"""FastAPI route handlers."""

import json
import traceback
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from providers.common import get_user_facing_error_message
from providers.exceptions import InvalidRequestError, ProviderError
from providers.logging_utils import build_request_summary, log_request_compact

from .dependencies import get_provider_for_type, get_settings
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .request_utils import get_token_count

router = APIRouter()


def _extract_model_from_auth_headers(raw_request: Request) -> str | None:
    """Extract model override from x-api-key or authorization header if prefixed with freecc:."""
    token = raw_request.headers.get("x-api-key")
    if not token:
        auth_header = raw_request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()

    if token and token.startswith("freecc:"):
        model_part = token[len("freecc:") :].strip()
        if model_part:
            return model_part
    return None


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    settings: Settings = Depends(get_settings),
):
    """Create a message (always streaming)."""

    try:
        if not request_data.messages:
            raise InvalidRequestError("messages cannot be empty")

        # Check for model override from auth token (e.g. launched via claude-pick or custom ANTHROPIC_AUTH_TOKEN)
        header_model = _extract_model_from_auth_headers(raw_request)
        if header_model:
            valid_providers = ("nvidia_nim", "open_router", "lmstudio")
            if "/" in header_model and header_model.split("/", 1)[0] in valid_providers:
                resolved_full = header_model
            else:
                resolved_full = f"{settings.provider_type}/{header_model}"
            request_data.resolved_provider_model = resolved_full
            request_data.model = Settings.parse_model_name(resolved_full)

        optimized = try_optimizations(request_data, settings)
        if optimized is not None:
            return optimized

        # Resolve provider from the model-aware mapping
        provider_type = Settings.parse_provider_type(
            request_data.resolved_provider_model or settings.model
        )
        provider = get_provider_for_type(provider_type)

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        log_request_compact(logger, request_id, request_data)

        input_tokens = get_token_count(
            request_data.messages, request_data.system, request_data.tools
        )
        return StreamingResponse(
            provider.stream_response(
                request_data,
                input_tokens=input_tokens,
                request_id=request_id,
            ),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except ProviderError:
        raise
    except Exception as e:
        logger.error(f"Error: {e!s}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=get_user_facing_error_message(e),
        ) from e


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest):
    """Count tokens for a request."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    with logger.contextualize(request_id=request_id):
        try:
            tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            summary = build_request_summary(request_data)
            summary["request_id"] = request_id
            summary["input_tokens"] = tokens
            logger.info("COUNT_TOKENS: {}", json.dumps(summary))
            return TokenCountResponse(input_tokens=tokens)
        except Exception as e:
            logger.error(
                "COUNT_TOKENS_ERROR: request_id={} error={}\n{}",
                request_id,
                get_user_facing_error_message(e),
                traceback.format_exc(),
            )
            raise HTTPException(
                status_code=500, detail=get_user_facing_error_message(e)
            ) from e


@router.get("/")
async def root(settings: Settings = Depends(get_settings)):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/stop")
async def stop_cli(request: Request):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
