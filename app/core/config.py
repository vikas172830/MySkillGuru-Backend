from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Secrets that must never sign real tokens. The default below is deliberately
# in this set, so a production deploy that forgets JWT_SECRET_KEY fails fast
# at startup instead of accepting forgeable tokens.
_WEAK_JWT_SECRETS = {
    "", "change-me-in-production", "changeme", "change_me", "secret",
    "your-secret-key", "dev", "test", "jwt-secret",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MONGODB_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "myskillguru"  # never the LMS database ("nexus"/"lms_evaluation")

    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_DAYS: int = 1
    # Cookies are shared across ports on the same host (cookie scope ignores
    # port), so this name MUST differ from the LMS's "access_token_cookie".
    JWT_COOKIE_NAME: str = "myskillguru_access_token_cookie"

    ENV: str = "development"
    CORS_ORIGINS: str = "http://localhost:3001"  # MySkillGuru frontend port

    GEMINI_API_KEY: str = ""
    IMAGEKIT_PUBLIC_KEY: str = ""
    IMAGEKIT_PRIVATE_KEY: str = ""
    IMAGEKIT_URL_ENDPOINT: str = ""

    ANTHROPIC_API_KEY: str = ""
    REDIS_URL: str = "redis://localhost:6379"
    # Prepended to EVERY Redis key (rate limits, revocation, account-state
    # cache, job state) so this app can never collide with another app
    # sharing the same Redis server/database.
    REDIS_KEY_PREFIX: str = "myskillguru:"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    # Own collection — the Qdrant cluster may be shared with the LMS, whose
    # collection is "lms_course_material_chunks". Never reuse that name.
    QDRANT_COLLECTION: str = "myskillguru_course_material_chunks"

    # Rate limiting (see app/core/rate_limit.py). All windows are 60s.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = 200          # per IP, applied to every request
    RATE_LIMIT_AI_PER_MINUTE: int = 10               # per user, on single-action AI endpoints
    RATE_LIMIT_BULK_GRADING_PER_MINUTE: int = 60      # per user, /evaluate-answer-script only —
    # its "Evaluate All" button fires one request per ungraded answer script at once
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10              # per IP, on public unauthenticated endpoints

    # SSRF guard (see app/utils/net.py). Comma-separated extra hostnames the
    # server may fetch from, on top of the ImageKit host (always allowed).
    # Leave empty to rely on the public-IP-only check alone.
    OUTBOUND_FETCH_ALLOWED_HOSTS: str = ""

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @property
    def outbound_fetch_allowed_hosts_list(self) -> list[str]:
        hosts = [h.strip().lower() for h in self.OUTBOUND_FETCH_ALLOWED_HOSTS.split(",") if h.strip()]
        endpoint = self.IMAGEKIT_URL_ENDPOINT.strip()
        if endpoint:
            from urllib.parse import urlparse

            host = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}").hostname
            if host:
                hosts.append(host.lower())
        return hosts

    @model_validator(mode="after")
    def _reject_weak_jwt_secret_in_production(self) -> "Settings":
        if self.is_production and (
            self.JWT_SECRET_KEY.strip().lower() in _WEAK_JWT_SECRETS
            or len(self.JWT_SECRET_KEY) < 32
        ):
            raise ValueError(
                "JWT_SECRET_KEY must be a strong random value (>= 32 chars) when ENV=production. "
                'Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        return self

    @model_validator(mode="after")
    def _reject_wildcard_cors_with_credentials(self) -> "Settings":
        # CORSMiddleware is mounted with allow_credentials=True; "*" would
        # silently become an any-origin credentialed policy.
        if "*" in self.cors_origins_list:
            raise ValueError(
                'CORS_ORIGINS must be an explicit list of origins — "*" is not allowed '
                "because the API sends credentials (cookies)."
            )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
