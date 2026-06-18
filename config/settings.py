import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Agency identity
    AGENCY_NAME: str = os.getenv("AGENCY_NAME", "Emergency Management Office")

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8092))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Output root — individual events write to output/{timestamp}/
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required")
        return True


settings = Settings()
