import asyncio
import logging
import httpx

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def provider_error(error: Exception) -> HTTPException:
    if isinstance(error, HTTPException):
        return error
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return HTTPException(504, "Hết thời gian xử lý. Hãy thử lại với tài liệu ngắn hơn.")
    if str(code) == "429":
        return HTTPException(429, "Dịch vụ AI hoặc Qdrant đã hết hạn mức. Vui lòng thử lại sau.")
    # Never include provider error bodies: they can contain credentials or document text.
    logger.warning("Cloud provider failed (%s, status=%s)", type(error).__name__, code)
    return HTTPException(502, "Dịch vụ AI hoặc Qdrant tạm thời không phản hồi. Vui lòng thử lại.")
