from whatsthatfish.src.database.base import Base
from whatsthatfish.src.database.config import (
    get_async_engine,
    get_async_session_factory,
    get_database_url,
    get_engine,
    get_session_factory,
)
from whatsthatfish.src.database.models import (  # noqa: F401 — registers all mappers with Base.metadata
    InatTaxa,
    InatFilteredObservations,
    LilaAnnotations,
    LilaCollectedImages,
    InatCaptureContext,
    InatImageQuality,
    LilaImageQuality,
    LilaYolo,
    SuccessfulUploads,
)

__all__ = [
    "Base",
    "get_database_url",
    "get_engine",
    "get_async_engine",
    "get_session_factory",
    "get_async_session_factory",
    "InatTaxa",
    "InatFilteredObservations",
    "LilaAnnotations",
    "LilaCollectedImages",
    "InatCaptureContext",
    "InatImageQuality",
    "LilaImageQuality",
    "LilaYolo",
    "SuccessfulUploads",
]
