import sys
from ..config import get_config, _get_logger
import logging
import os
from pathlib import Path
from .gcs_client import GCSClient
from .inaturalist_dataset import INaturalistDataset
from .download_lila import LilaDataset
import asyncio


class DataFactory:

    def __init__(self,
                 datasets: str,):
        self.config = get_config()
        self.datasets = datasets
        self.base_path = os.getcwd()
        self.data_path = Path(__file__).parents[2] / "data"
        self.logger = _get_logger("DataFactory")
        self.gcs = GCSClient(self.config.gcs)
        self.inaturalist_dataset = INaturalistDataset(self.config.s3)
        self.lila_dataset = LilaDataset(self.gcs)

    async def run_data_extraction(self):

        self.logger.info("Starting data extraction")
        self.logger.debug("Initializing classes.")

        print(self.data_path)
        await self.inaturalist_dataset.retrieve_s3_data_by_bucket(str(self.data_path))
        # await self.lila_dataset.extract_lila_images()
        # self.lila_dataset.download_coco_json()

        """
        if self.datasets == "all":
            

        else if datasets == "classification":
            for key, value in self.inaturalist_dataset.s3_config.datasets.items():
                self.inaturalist_dataset.retrieve_s3_data_by_bucket(self.data_path)
            self.inaturalist_dataset.retrieve_s3_data_by_bucket()



        else if datasets == "detection":


        else:
            return "Invalid dataset"


        """


async def run_data_factory():
    datafactory = DataFactory("all")
    return await datafactory.run_data_extraction()


def main():
    return asyncio.run(run_data_factory())

if __name__ == "__main__":
    main()





