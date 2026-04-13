"""Shared retry decorators for all external service interactions.

Three retry domains:
    db_retry   — Postgres: OperationalError, InterfaceError
    s3_retry   — AWS S3 (boto3): ClientError, ConnectionError, ConnectionClosedError, tarfile.ReadError
    gcs_retry  — GCS (sync google-cloud-storage): ClientError, ServerError
    transfer_retry — async aiohttp + GCS: 429/5xx HTTP errors, connector errors, GCS transient errors

Each decorator uses exponential backoff with 5 attempts.
"""

import tarfile

import aiohttp
from botocore.exceptions import (
    ClientError as BotoClientError,
    ConnectionError as BotoConnectionError,
    ConnectionClosedError,
)
from google.api_core.exceptions import (
    ClientError as GCSClientError,
    ServerError as GCSServerError,
    ServiceUnavailable,
    TooManyRequests,
    GoogleAPICallError,
)
from sqlalchemy.exc import OperationalError, InterfaceError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import _get_logger

logger = _get_logger("retry")


def _log_retry(retry_state):
    logger.warning(
        f"[{retry_state.fn.__qualname__}] "
        f"Attempt {retry_state.attempt_number} failed: "
        f"{retry_state.outcome.exception()}. "
        f"Retrying in {retry_state.next_action.sleep:.1f}s..."
    )


# ── Database (Postgres via SQLAlchemy) ─────────────────────────────────
db_retry = retry(
    retry=(
        retry_if_exception_type(OperationalError)
        | retry_if_exception_type(InterfaceError)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)

# ── AWS S3 (boto3 sync) ───────────────────────────────────────────────
s3_retry = retry(
    retry=(
        retry_if_exception_type(BotoClientError)
        | retry_if_exception_type(BotoConnectionError)
        | retry_if_exception_type(ConnectionClosedError)
        | retry_if_exception_type(tarfile.ReadError)
    ),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)

# ── GCS (sync google-cloud-storage) ───────────────────────────────────
gcs_retry = retry(
    retry=(
        retry_if_exception_type(GCSClientError)
        | retry_if_exception_type(GCSServerError)
    ),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)


# ── Async transfer (aiohttp + gcloud-aio-storage) ─────────────────────
def _transfer_retry_predicate(exc: BaseException) -> bool:
    """Retry 429/5xx HTTP errors, connector errors, and GCS transient errors."""
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (429, 500, 502, 503, 504)
    if isinstance(exc, aiohttp.ClientConnectorError):
        return True
    if isinstance(exc, (ServiceUnavailable, TooManyRequests, GoogleAPICallError)):
        return True
    return False


transfer_retry = retry(
    retry=retry_if_exception(_transfer_retry_predicate),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)
