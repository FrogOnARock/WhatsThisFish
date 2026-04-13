"""
Shared fixtures for integration tests.

Requires a running test Postgres instance:
    docker compose -f docker-compose.test.yml up -d

The session_factory fixture creates all tables fresh for each test,
then rolls back via truncation after each test for isolation.
"""

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from whatsthatfish.src.database.base import Base

TEST_DATABASE_URL = "postgresql://test:test@localhost:5433/wtf_test"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def engine():
    """Create a single engine for the entire test session.

    Creates all tables on first use. The engine is shared across tests
    for efficiency — isolation is handled per-test via truncation.
    """
    eng = create_engine(TEST_DATABASE_URL, echo=False)

    # Verify connection before running any tests
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(
            f"Test Postgres not available at {TEST_DATABASE_URL}. "
            f"Run: docker compose -f docker-compose.test.yml up -d\n"
            f"Error: {e}"
        )

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    """Provide a session factory pointing at the test DB.

    After each test, truncates all tables to ensure isolation.
    TRUNCATE CASCADE handles FK ordering automatically.
    """
    factory = sessionmaker(bind=engine)
    yield factory

    # Teardown: truncate all tables for clean slate
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
        conn.commit()


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory containing parquet files."""
    return FIXTURES_DIR
