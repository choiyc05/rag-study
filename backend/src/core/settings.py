from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """.env 값을 읽어 타입 검증까지 해주는 설정 객체."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GEMINI_API_KEY: str
    # .env에 CORS_ORIGINS='["http://localhost:3000"]' 형태(JSON)로 넣으면 리스트로 파싱됨
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]  # 개발 편의상 모든 도메인 허용. 배포 시에는 실제 도메인만 허용하도록 수정.


@lru_cache
def get_settings() -> Settings:
    """앱 전체에서 Settings 인스턴스를 하나만 쓰도록 캐싱.

    .env를 매번 다시 읽지 않게 하고, FastAPI의 Depends(get_settings)로도 주입 가능.
    """
    return Settings()
