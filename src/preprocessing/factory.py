from enum import Enum
from pathlib import Path
from sqlalchemy import inspect
import asyncio

from .score_runner import ScoreRunner, ContextRunner
from .annotation_conversion import AnnotationConverter
from .score_runner import ScoringProgressTracker
from dataclasses import dataclass
from ..database.models import InatCaptureContext, InatImageQuality, LilaImageQuality
from ..database import get_session_factory
from ..config import get_config
from ..etl.gcs_client import GCSClient


@dataclass
class Dataset(str, Enum):
    ALL = "all"
    SCORING = "scoring"
    ANN_CONV = "annotation"
    CONTEXT = "context"

class PreProcessingFactory:
    def __init__(self,
                 type: Dataset):

        self.data_path = Path(__file__).parents[1] / "data" / "preprocessing"
        self.session_factory = get_session_factory()
        self.config = get_config()
        self.gcs_config = self.config.gcs
        self.gcs_client = GCSClient(self.gcs_config)
        self.type = type

    def _dest_table(self, dataset: str, runner: str):
        if dataset == "lila":
            return LilaImageQuality
        if dataset == "inat":
            return InatImageQuality if runner == "scoring" else InatCaptureContext
        raise ValueError(f"Unknown dataset: {dataset!r}")

    def _load_score_runner(self, dataset: str, source: str):

        dest_table = self._dest_table(dataset, runner="scoring")

        tracker = ScoringProgressTracker(
            data_path=str(self.data_path),
            source=dataset,
            session_factory=self.session_factory,
            dest_table=dest_table,
            pk = str(inspect(dest_table).mapper.primary_key[0].name),
        )

        return ScoreRunner(
            gcs_config=self.gcs_config,
            session=self.session_factory,
            progress_tracker=tracker,
            dataset=dataset)

    def _load_context_runner(self, dataset: str, source: str):

        dest_table = self._dest_table(dataset, runner="context")

        tracker = ScoringProgressTracker(
            data_path=str(self.data_path),
            source=source,
            session_factory=self.session_factory,
            dest_table=dest_table,
            pk = str(inspect(dest_table).mapper.primary_key[0].name),
        )

        return ContextRunner(
            gcs_config=self.gcs_config,
            session=self.session_factory,
            progress_tracker=tracker,
            dataset=dataset)



    def _load_annotation_runner(self):
        return AnnotationConverter(
            session_factory=self.session_factory
        )

    async def run(self):

        if self.type == "all":
            await self._load_context_runner(dataset="inat", source="inat_context").run()
            await self._load_score_runner(dataset="inat", source="inat_scoring").run()
            await self._load_score_runner(dataset="lila", source="lila_scoring").run()
            self._load_annotation_runner().run()

        elif self.type == "scoring":
            await self._load_score_runner(dataset="inat", source="inat_scoring").run()
            await self._load_score_runner(dataset="lila", source="lila_scoring").run()

        elif self.type == "context":
            await self._load_context_runner(dataset="inat", source="inat_context").run()

        elif self.type == "annotation":
            self._load_annotation_runner().run()

        else:
            raise ValueError("Error")


if __name__ == '__main__':
    asyncio.run(PreProcessingFactory(type=Dataset.ALL).run())