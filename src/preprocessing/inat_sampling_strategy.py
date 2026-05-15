import numpy as np
from sqlalchemy import select, func, desc
from dotenv import load_dotenv
import cuml.accel
cuml.accel.install()
from sklearn.cluster import HDBSCAN

from ..database import InatImageQuality
from ..database.config import get_session_factory
from ..database.models import InatFilteredObservations, InatClipContext
from ..config import _get_logger

logger = _get_logger(__name__)


def hdbscan_clustering(rows):
    photo_uuids = []
    lat_lon_pairs = []
    for row in rows:
        photo_uuids.append(row.photo_uuid)
        lat_lon_pairs.append([row.latitude, row.longitude])
    lat_lon = np.array(lat_lon_pairs, dtype=np.float32)
    coords_radians = np.radians(lat_lon)

    logger.info(f"The length of our radians coordinates is {len(coords_radians)}")
    logger.info("Clustering leveraging HDBSCAN.")
    hdb = HDBSCAN(min_cluster_size=1000, metric='haversine', copy=True)
    labels = hdb.fit_predict(coords_radians)
    clustered_pairs = list(zip(photo_uuids, labels))

    logger.info(f"Length of clustered pairs {len(clustered_pairs)}")
    logger.info(f"\nFirst five pairs: \n{clustered_pairs[:5]}")

    return clustered_pairs


def retrieve_sampled():
    logger.info("Retrieving lat, lon pairs for iNaturalist Observations.")

    stmt = (
        select(
            InatClipContext.photo_uuid,
            InatFilteredObservations.photo_id,
            InatFilteredObservations.extension,
            InatFilteredObservations.taxon_id,
            InatFilteredObservations.latitude,
            InatFilteredObservations.longitude,
            InatImageQuality.uiqm,
            InatClipContext.is_underwater,
            func.dense_rank().over(partition_by=InatFilteredObservations.taxon_id, order_by=desc(InatImageQuality.uiqm)).label("uiqm_rank"),
            func.count().over(partition_by=InatFilteredObservations.taxon_id).label("taxon_count"))
        .join(InatFilteredObservations, InatClipContext.photo_uuid == InatFilteredObservations.photo_uuid)
        .join(InatImageQuality, InatClipContext.photo_uuid == InatImageQuality.photo_uuid)
        .where(InatClipContext.is_underwater == 1).where(InatImageQuality.uiqm.isnot(None))
    )
    cte = stmt.cte("ranked")
    outer = select(
        cte.c.photo_uuid,
        func.concat(cte.c.photo_id, '.', cte.c.extension).label("filename"),
        cte.c.longitude,
        cte.c.latitude,
        cte.c.taxon_id,
        cte.c.uiqm_rank,
        cte.c.uiqm,
        cte.c.is_underwater,
    ).where(cte.c.taxon_count > 300, cte.c.uiqm_rank <= 300)

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(outer).all()

    logger.info("Pairs retrieved.")
    return rows





if __name__ == '__main__':
    load_dotenv()
    rows = retrieve_sampled()
    clustered_pairs = hdbscan_clustering(rows)














