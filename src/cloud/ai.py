import asyncio
import json
import re

from fastapi import HTTPException

from .config import CloudSettings, EMBEDDING_DIMENSIONS


def parse_json(value: str):
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.I)
    try:
        return json.loads(value)
    except (ValueError, TypeError) as error:
        raise HTTPException(502, "AI trả về dữ liệu không hợp lệ. Vui lòng thử lại.") from error


class GeminiAI:
    def __init__(self, settings: CloudSettings):
        from google import genai
        from google.genai import types

        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key,
                                   http_options=types.HttpOptions(timeout=90_000,
                                       retry_options=types.HttpRetryOptions(attempts=1)))

    async def close(self):
        await self.client.aio.aclose()

    async def generate(self, prompt: str, *, json_output=False) -> str:
        from google.genai import types

        response = await self.client.aio.models.generate_content(
            model=self.settings.model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, max_output_tokens=8192,
                response_mime_type="application/json" if json_output else "text/plain"))
        if not response.text or not response.text.strip():
            raise HTTPException(502, "AI chưa trả về nội dung. Vui lòng thử lại.")
        return response.text.strip()

    async def embed(self, text: str, *, title: str | None = None) -> list[float]:
        from google.genai import types

        contents = f"title: {title} | text: {text}" if title is not None else f"task: search result | query: {text}"
        result = await self.client.aio.models.embed_content(
            model=self.settings.embedding_model, contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS))
        embeddings = result.embeddings or []
        if len(embeddings) != 1 or not embeddings[0].values or len(embeddings[0].values) != EMBEDDING_DIMENSIONS:
            raise HTTPException(502, "Dịch vụ embedding trả về vector không hợp lệ.")
        return embeddings[0].values

    async def embed_documents(self, chunks: list[dict]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(4)

        async def one(chunk):
            async with semaphore:
                return await self.embed(chunk["content"], title=chunk["filename"])

        # Each SDK request embeds one chunk; never aggregate contents into one vector.
        tasks = [asyncio.create_task(one(chunk)) for chunk in chunks]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
