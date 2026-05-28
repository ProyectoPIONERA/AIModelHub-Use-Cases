"""Simple GTFS-RT collector utilities.

This module provides small helpers to snapshot TripUpdates and VehiclePositions
feeds and store them under `data/mobility-datasets/GTFS_RT/` with a
timestamped filename. The collector is intentionally lightweight: it fetches
the raw response bytes and saves them for later parsing by preprocessing.

Usage example:
    from src.utils.mobility.collector import snapshot_feeds
    snapshot_feeds()

"""
import os
import time
from datetime import datetime
import requests

RT_DIR = "data/mobility-datasets/GTFS_RT"


def ensure_rt_dir():
    os.makedirs(RT_DIR, exist_ok=True)


def snapshot_url(url: str, name_prefix: str):
    ensure_rt_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{ts}_{name_prefix}"
    path = os.path.join(RT_DIR, filename)

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    # Try to detect content type: JSON vs protobuf
    content_type = r.headers.get("Content-Type", "")
    if "application/json" in content_type or r.text.strip().startswith("{"):
        path = path + ".json"
        with open(path, "w", encoding="utf8") as fh:
            fh.write(r.text)
    else:
        # protobuf / binary
        path = path + ".pb"
        with open(path, "wb") as fh:
            fh.write(r.content)

    return path


def snapshot_feeds(trip_updates_url: str = "https://api.control.optibus.co/opendata/v1/gtfs-rt/trip-updates?uid=c-5cfcd2d1",
                   vehicle_positions_url: str = "https://api.control.optibus.co/opendata/v1/gtfs-rt/vehicle-positions?uid=c-5cfcd2d1"):
    """Fetch both feeds and save snapshots to disk.

    Returns a tuple with the saved file paths.
    """
    tu_path = snapshot_url(trip_updates_url, "trip_updates")
    vp_path = snapshot_url(vehicle_positions_url, "vehicle_positions")
    return tu_path, vp_path


if __name__ == "__main__":
    print("Snapshotting feeds...")
    print(snapshot_feeds())
