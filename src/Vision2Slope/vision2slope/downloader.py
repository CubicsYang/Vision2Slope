"""
Google Street View downloader using streetlevel library.

Downloads panoramic images based on coordinates, producing a metadata CSV
with pano_id, heading, lat, lon that can be used with the Vision2Slope pipeline.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from streetlevel import streetview
except ImportError:
    streetview = None

logger = logging.getLogger(__name__)


class GSVDownloader:
    """Download Google Street View panoramas based on coordinates."""

    def __init__(
        self,
        output_dir: str,
        zoom: int = 5,
        search_radius: int = 50,
        delay: float = 0.5,
    ):
        """
        Args:
            output_dir: Directory to save downloaded panoramas
            zoom: Image zoom level (0=lowest, 5=highest)
            search_radius: Search radius in meters for finding panoramas
            delay: Delay between requests in seconds to avoid rate limiting
        """
        if streetview is None:
            raise ImportError(
                "streetlevel is required for GSV downloading. "
                "Install it with: pip install streetlevel"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zoom = zoom
        self.search_radius = search_radius
        self.delay = delay

    def download_from_coords(
        self,
        coords: List[Tuple[float, float]],
        edge_bearings: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Find and download GSV panoramas near given coordinates.

        Args:
            coords: List of (lat, lon) tuples (e.g. sampled points along OSM edges)
            edge_bearings: Optional list of OSM edge bearings (u→v direction)
                corresponding to each coordinate. If provided, included in output CSV.

        Returns:
            DataFrame with columns: pano_id, lat, lon, heading, edge_bearing (if provided)
            Also saves the DataFrame as metadata.csv in output_dir.
        """
        records = []
        downloaded_ids = set()

        for i, (lat, lon) in enumerate(tqdm(coords, desc="Downloading GSV panoramas")):
            try:
                pano = streetview.find_panorama(
                    lat, lon, radius=self.search_radius
                )

                if pano is None:
                    logger.warning(f"No panorama found near ({lat}, {lon})")
                    continue

                if pano.id in downloaded_ids:
                    logger.debug(f"Skipping duplicate pano {pano.id}")
                    # Still record the metadata for this point
                    record = {
                        "pano_id": pano.id,
                        "lat": pano.lat,
                        "lon": pano.lon,
                        "heading": pano.heading or 0.0,
                    }
                    if edge_bearings is not None:
                        record["edge_bearing"] = edge_bearings[i]
                    records.append(record)
                    continue

                # Download panorama
                output_path = str(self.output_dir / f"{pano.id}.jpg")
                streetview.download_panorama(pano, output_path, zoom=self.zoom)
                downloaded_ids.add(pano.id)

                record = {
                    "pano_id": pano.id,
                    "lat": pano.lat,
                    "lon": pano.lon,
                    "heading": pano.heading or 0.0,
                }
                if edge_bearings is not None:
                    record["edge_bearing"] = edge_bearings[i]
                records.append(record)

                logger.debug(
                    f"Downloaded {pano.id} at ({pano.lat}, {pano.lon}), "
                    f"heading={pano.heading}"
                )

            except Exception as e:
                logger.error(f"Failed to download panorama at ({lat}, {lon}): {e}")
                continue

            if self.delay > 0:
                time.sleep(self.delay)

        metadata_df = pd.DataFrame(records)

        # Save metadata CSV
        csv_path = self.output_dir / "metadata.csv"
        metadata_df.to_csv(csv_path, index=False)
        logger.info(
            f"Downloaded {len(downloaded_ids)} panoramas, "
            f"metadata saved to {csv_path}"
        )

        return metadata_df

    def download_from_dataframe(
        self,
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        edge_bearing_col: Optional[str] = "edge_bearing",
    ) -> pd.DataFrame:
        """
        Download GSV panoramas from a DataFrame with coordinate columns.

        Args:
            df: DataFrame with lat/lon columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            edge_bearing_col: Name of edge bearing column (None to skip)

        Returns:
            DataFrame with pano metadata
        """
        coords = list(zip(df[lat_col], df[lon_col]))
        edge_bearings = (
            df[edge_bearing_col].tolist()
            if edge_bearing_col and edge_bearing_col in df.columns
            else None
        )
        return self.download_from_coords(coords, edge_bearings)
