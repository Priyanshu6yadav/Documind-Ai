from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b"
    embedding_model: str = "batiai/qwen3-embedding:0.6b-q6"
    ollama_num_ctx: int = 4096

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()