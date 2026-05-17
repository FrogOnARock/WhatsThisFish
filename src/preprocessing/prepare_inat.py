from typing import Any

import numpy as np
from sqlalchemy import select, func, desc, insert
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from cuml.cluster import KMeans as cuKMeans
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from ..database.config import get_session_factory
from ..database.models import InatClipContext, InatFilteredObservations, InatImageQuality, InatTaxa, InatClassificationDataset
from ..config import _get_logger
from ..retry import db_retry


logger = _get_logger(__name__)

class InatPreparation:
    def __init__(self,
                 session_factory: sessionmaker,
                 kmeans_search: bool = True):
        self.session_factory = session_factory()
        self.kmeans_search = kmeans_search


    def _plot_kmeans_search(self, results: dict, best_k: int):
        ks = sorted(results.keys())
        silhouettes = [results[k]["silhouette"] for k in ks]
        inertias = [results[k]["inertia"] for k in ks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(ks, silhouettes, marker='o')
        ax1.axvline(x=best_k, color='r', linestyle='--', label=f'best k={best_k}')
        ax1.set_xlabel("k")
        ax1.set_ylabel("silhouette score")
        ax1.set_title("Silhouette vs k")
        ax1.legend()

        ax2.plot(ks, inertias, marker='o')
        ax2.axvline(x=best_k, color='r', linestyle='--', label=f'best k={best_k}')
        ax2.set_xlabel("k")
        ax2.set_ylabel("inertia")
        ax2.set_title("Elbow vs k")
        ax2.legend()

        plt.tight_layout()
        plt.savefig('./kmeans_search/kmeans_search.png')
        plt.close()


    def _visualize_clusters(self, coords_radians: np.ndarray, labels: np.ndarray, k: int, filename: str):
        cmap = plt.get_cmap('tab20' if k <= 20 else 'hsv')
        fig, ax = plt.subplots(figsize=(16, 8))

        patches = []
        patch_colors = []

        for cluster_id in range(k):
            pts = coords_radians[labels == cluster_id]
            if len(pts) < 3:
                continue
            color = cmap(cluster_id / k)
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                polygon_xy = np.column_stack([hull_pts[:, 1], hull_pts[:, 0]])  # lon, lat
                patches.append(Polygon(polygon_xy, closed=True))
                patch_colors.append(color)
            except Exception:
                pass

            centroid_lon = np.mean(pts[:, 1])
            centroid_lat = np.mean(pts[:, 0])
            ax.text(centroid_lon, centroid_lat, str(cluster_id),
                    ha='center', va='center', fontsize=7, fontweight='bold', color='black')

        collection = PatchCollection(patches, facecolor=[(*c[:3], 0.3) for c in patch_colors],
                                     edgecolor=[(*c[:3], 0.9) for c in patch_colors], linewidth=1)
        ax.add_collection(collection)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi / 2, np.pi / 2)
        ax.set_xlabel("longitude (radians)")
        ax.set_ylabel("latitude (radians)")
        ax.set_title(f"KMeans clusters (k={k})")
        plt.tight_layout()
        plt.savefig(f'./kmeans_search/{filename}')
        plt.close()


    def _write_kmeans_log(self, results: dict):
        with open('./kmeans_search/kmeans_log.txt', 'w') as f:
            for k in sorted(results.keys()):
                f.write(f"k={k}, silhouette={results[k]['silhouette']:.6f}, inertia={results[k]['inertia']:.2f}\n")


    def run_kmeans_search(self, coords_radians, k_range: range = range(5, 100, 5)) -> int:
        """
        Sweep k, scoring each with silhouette (higher = better separation) and
        inertia (elbow method — look for the kink where adding k stops helping).
        Returns best_k by silhouette; elbow plot gives visual confirmation.
        """
        results: dict[int, dict] = {}

        logger.info(f"Sweeping k in {list(k_range)}.")
        for k in k_range:
            km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
            labels = km.fit_predict(coords_radians)
            sil = silhouette_score(coords_radians, labels, sample_size=10_000, random_state=42)
            results[k] = {"silhouette": sil, "inertia": km.inertia_}
            logger.info(f"k={k} silhouette={sil:.4f} inertia={km.inertia_:.2f}")

        best_k = max(results, key=lambda k: results[k]["silhouette"])
        self._write_kmeans_log(results)
        self._plot_kmeans_search(results, best_k)

        best_labels = MiniBatchKMeans(n_clusters=best_k, n_init=3, random_state=42).fit_predict(coords_radians)
        self._visualize_clusters(coords_radians, best_labels, best_k, f'clusters_subsample_k{best_k}.png')
        logger.info(f"Search complete. best_k={best_k} silhouette={results[best_k]['silhouette']:.4f}")

        return best_k


    def cu_fit(self, data, k: int) -> Any:
        """GPU KMeans for the final full-dataset fit."""
        km = cuKMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit_predict(data)
        return km.labels_


    def _subsample(self, coords_radians: np.ndarray, n: int = 10000) -> np.ndarray:
        """
        Return a representative subsample of coords_radians for hyperparameter search.
        """

        logger.info(f"Creating a subsample of {n} data points.")
        min_lat, max_lat, dev_lat = np.min(coords_radians[:, 0]), np.max(coords_radians[:, 0]), np.std(coords_radians[:, 0])
        min_lon, max_lon, dev_lon = np.min(coords_radians[:, 1]), np.max(coords_radians[:, 1]), np.std(coords_radians[:, 1])

        step_lat = 0
        lat_bins = []
        while True:
            current_lat = min_lat + step_lat
            lat_bins.append(current_lat)
            step_lat += dev_lat
            if current_lat + step_lat > max_lat:
                lat_bins.append(max_lat)
                break

        step_lon = 0
        lon_bins = []
        while True:
            current_lon = min_lon + step_lon
            lon_bins.append(current_lon)
            step_lon += dev_lon
            if current_lon + step_lon > max_lon:
                lon_bins.append(max_lon)
                break

        lat_bins_arr = np.array(lat_bins)
        lon_bins_arr = np.array(lon_bins)

        logger.info(f"Binning coordinate radians.")
        coords_radians_bin_lat = np.digitize(coords_radians[:, 0], lat_bins_arr)
        coords_radians_bin_lon = np.digitize(coords_radians[:, 1], lon_bins_arr)

        cell_ids = coords_radians_bin_lat * len(lon_bins_arr) + coords_radians_bin_lon

        unique_cells = np.unique(cell_ids)
        samples_per_cell = max(1, n // len(unique_cells))

        logger.info("Selecting subsample.")
        selected = []
        for cell in unique_cells:
            idx = np.where(cell_ids == cell)[0]
            k = min(len(idx), samples_per_cell)
            selected.append(np.random.choice(idx, size=k, replace=False))

        selected_idx = np.concatenate(selected)
        if len(selected_idx) > n:
            selected_idx = np.random.choice(selected_idx, size=n, replace=False)

        logger.info(f"Subsample selected of size: {len(selected_idx)}")
        return coords_radians[selected_idx]


    def kmeans_clustering(self, rows, search: bool):
        lat_lon_pairs = [[row["latitude"], row["longitude"]] for row in rows]
        lat_lon = np.array(lat_lon_pairs, dtype=np.float32)
        coords_radians = np.radians(lat_lon)

        if search:
            logger.info(f"Running k search on subsample of {len(coords_radians)} points.")
            search_coords = self._subsample(coords_radians)
            best_k = self.run_kmeans_search(search_coords)
            with open('./kmeans_search/best_k.txt', 'w') as f:
                f.write(f"{best_k}")
        else:
            with open('./kmeans_search/best_k.txt', 'r') as f:
                best_k = int(f.read())

        logger.info(f"Fitting cuKMeans k={best_k} on full dataset.")
        labels = self.cu_fit(coords_radians, best_k)
        self._visualize_clusters(coords_radians, np.array(labels), best_k, f'clusters_full_k{best_k}.png')

        for i, row in enumerate(rows):
            row["cluster"] = int(labels[i])
        return rows

    def split_taxa(self, row: dict[str, str]) -> dict[str, str]:
        split = row["ancestry"].split("/")
        row["species"], row["genus"], row["subfamily"] = row["taxon_id"], int(split[-1]), int(split[-2])
        return row

    @db_retry
    def retrieve_sampled(self):

        """
        Retrieve the relevant iNaturalist observations with:
        longitude and latitude for geographic clustering -> required for the train test split
        ancestry + taxon_id -> required to split into Subfamily, Genus, Species
        photo_id + extension -> together form the URL for GCP GET Request
        uiqm -> required for weighted sampling based on image quality
        is_underwater -> filter to ensure we're leveraging the underwater images as defined by CLIP

        results: rows dict containing data
        """

        logger.info("Retrieving lat, lon pairs for iNaturalist Observations.")

        stmt = (
            select(
                InatTaxa.ancestry,
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
            .join(InatTaxa, InatFilteredObservations.taxon_id == InatTaxa.taxon_id)
            .where(InatClipContext.is_underwater == 1).where(InatImageQuality.uiqm.isnot(None)).where(InatTaxa.rank == 'species').where(InatFilteredObservations.latitude.isnot(None) & InatFilteredObservations.longitude.isnot(None))
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
            cte.c.ancestry
        ).where(cte.c.taxon_count > 300, cte.c.uiqm_rank <= 300)

        session_factory = get_session_factory()
        with session_factory() as session:
            rows = session.execute(outer).all()

        logger.info("Pairs retrieved.")

        return_list = [{
            "photo_uuid": row.photo_uuid,
            "filename": row.filename,
            "longitude": row.longitude,
            "latitude": row.latitude,
            "taxon_id": row.taxon_id,
            "uiqm_rank": row.uiqm_rank,
            "uiqm": row.uiqm,
            "is_underwater": row.is_underwater,
            "ancestry": row.ancestry
        } for row in rows]

        return return_list

    @db_retry
    def load(self, rows) -> bool:
        with self.session_factory() as session:
            session.execute(insert(InatClassificationDataset).values(rows))
            session.commit()
        return True

    def run(self):
        load_dotenv()
        Path.mkdir(Path("./kmeans_search"), exist_ok=True)
        rows = self.retrieve_sampled()
        ancestry_incl_rows = [self.split_taxa(row) for row in rows]
        clustered_rows = self.kmeans_clustering(ancestry_incl_rows, search=False)
        self.load(clustered_rows)

