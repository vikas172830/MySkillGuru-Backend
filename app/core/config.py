from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MONGODB_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "lms_evaluation"

    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_DAYS: int = 1
    JWT_COOKIE_NAME: str = "access_token_cookie"

    ENV: str = "development"
    CORS_ORIGINS: str = "http://localhost:3000"

    GEMINI_API_KEY: str = ""
    IMAGEKIT_PUBLIC_KEY: str = ""
    IMAGEKIT_PRIVATE_KEY: str = ""
    IMAGEKIT_URL_ENDPOINT: str = ""

    ANTHROPIC_API_KEY: str = ""
    REDIS_URL: str = "redis://localhost:6379"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""

    # Rate limiting (see app/core/rate_limit.py). All windows are 60s.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = 200          # per IP, applied to every request
    RATE_LIMIT_AI_PER_MINUTE: int = 10               # per user, on single-action AI endpoints
    RATE_LIMIT_BULK_GRADING_PER_MINUTE: int = 60      # per user, /evaluate-answer-script only —
    # its "Evaluate All" button fires one request per ungraded answer script at once
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10              # per IP, on public unauthenticated endpoints

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
