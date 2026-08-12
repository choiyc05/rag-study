from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """.env 값을 읽어 타입 검증까지 해주는 설정 객체.

    필드명(GEMINI_API_KEY)과 .env 키 이름이 그대로 매칭된다.
    기본값을 주지 않은 필드는 .env에 없으면 앱 기동 시점에 에러가 나므로,
    "실행은 됐는데 키가 None이라 API 호출에서 터지는" 상황을 막아준다.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GEMINI_API_KEY: str


@lru_cache
def get_settings() -> Settings:
    """앱 전체에서 Settings 인스턴스를 하나만 쓰도록 캐싱.

    .env를 매번 다시 읽지 않게 하고, FastAPI의 Depends(get_settings)로도 주입 가능.
    """
    return Settings()
