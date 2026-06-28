import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings:
    PROJECT_ROOT: Path = PROJECT_ROOT

    # Agency identity
    AGENCY_NAME: str = os.getenv("AGENCY_NAME", "Emergency Management Office")

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Neon (Vercel Postgres)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Evo 1.0 model
    EVO_MODEL_VERSION: str = os.getenv("EVO_MODEL_VERSION", "evo1.2")
    EVO_MODEL_REPO: str = os.getenv("EVO_MODEL_REPO", "https://github.com/rvkshn-10/Evo")
    EVO_PREFER_OPENVINO: bool = os.getenv("EVO_PREFER_OPENVINO", "true").lower() == "true"
    EVO_HYBRID_MODE: bool = os.getenv("EVO_HYBRID_MODE", "true").lower() == "true"
    EVO_TIME_CATEGORIES: str = os.getenv("EVO_TIME_CATEGORIES", "Train Station")

    # CORS — comma-separated extra origins (e.g. Vercel preview URL)
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "")

    # Search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # PeopleSense FCUSD — GET occupancy database + POST event API (x-api-key)
    PEOPLESENSE_API_KEY: str = os.getenv("PEOPLESENSE_API_KEY", "placeholder")
    PEOPLESENSE_OCCUPANCY_URL: str = os.getenv(
        "PEOPLESENSE_OCCUPANCY_URL",
        "https://w8bdwhaps0.execute-api.us-west-2.amazonaws.com/v1/occupancy",
    )
    PEOPLESENSE_CACHE_TTL_SECONDS: int = int(os.getenv("PEOPLESENSE_CACHE_TTL_SECONDS", "60"))
    PEOPLESENSE_EVENT_URL: str = os.getenv(
        "PEOPLESENSE_EVENT_URL",
        "https://k0i6cg0ob2.execute-api.us-west-2.amazonaws.com/prod/event",
    )
    # Legacy placeholder — unused when PEOPLESENSE_EVENT_URL is set
    PEOPLESENSE_BASE_URL: str = os.getenv(
        "PEOPLESENSE_BASE_URL", "https://api.peoplesense.ai"
    )

    # NOAA / NWS
    NWS_USER_AGENT: str = os.getenv(
        "NWS_USER_AGENT", "(emergency-management-office-ai, contact@example.com)"
    )
    DEFAULT_ALERT_AREA: str = os.getenv("DEFAULT_ALERT_AREA", "CA")
    DEFAULT_MAP_LAT: float = float(os.getenv("DEFAULT_MAP_LAT", "38.58"))
    DEFAULT_MAP_LON: float = float(os.getenv("DEFAULT_MAP_LON", "-121.30"))

    # USGS earthquake early warning (public feeds; ShakeAlert XML requires USGS partnership)
    USGS_EEW_MIN_MAGNITUDE: float = float(os.getenv("USGS_EEW_MIN_MAGNITUDE", "4.0"))

    # NASA FIRMS wildfire hotspots (free MAP_KEY — optional)
    NASA_FIRMS_MAP_KEY: str = os.getenv("NASA_FIRMS_MAP_KEY", "")

    # PeopleSense auto-deployment
    PEOPLESENSE_AUTO_DEPLOY: bool = os.getenv("PEOPLESENSE_AUTO_DEPLOY", "true").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8092))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Output root — individual events write to output/{timestamp}/
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")

    @classmethod
    def validate(cls, *, require_agent_keys: bool = True):
        if require_agent_keys:
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for the agent pipeline")
            if not cls.TAVILY_API_KEY:
                raise ValueError("TAVILY_API_KEY is required for the agent pipeline")
    def cors_origins(self) -> list[str]:
        base = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8092",
            "http://127.0.0.1:8092",
        ]
        if self.CORS_ORIGINS:
            base.extend(o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip())
        return base


settings = Settings()
