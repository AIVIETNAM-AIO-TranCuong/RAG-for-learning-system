from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_UPLOAD_BYTES = 4_000_000
OPERATION_SECONDS = 240
EMBEDDING_DIMENSIONS = 768


class CloudSettings(BaseSettings):
    gemini_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    model: str = Field("gemini-3.5-flash-lite", validation_alias="CLOUD_GEMINI_MODEL")
    embedding_model: str = Field("gemini-embedding-2", validation_alias="CLOUD_EMBEDDING_MODEL")
    collection: str = Field("documents_gemini_embedding2_768", validation_alias="CLOUD_COLLECTION")
    model_config = SettingsConfigDict(env_file=(".env", ".env.cloud"), extra="ignore")

    def missing(self) -> list[str]:
        return [name.upper() for name in ("gemini_api_key", "qdrant_url", "qdrant_api_key")
                if not getattr(self, name).strip()]
