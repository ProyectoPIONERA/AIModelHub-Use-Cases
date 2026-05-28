import os
import pandas as pd
from datetime import datetime
from typing import List, Dict
from google.protobuf.json_format import MessageToDict
from google.transit.gtfs_realtime_pb2 import FeedMessage
import numpy as np
from shapely.geometry import LineString,Point
from sklearn.model_selection import GroupShuffleSplit

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def gtfs_time_to_seconds(t):
    """
    GTFS HH:MM:SS → seconds
    supports >24h
    """
    if pd.isna(t):
        return np.nan

    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def build_static_segments(
    stop_times_df: pd.DataFrame,
    stops_df: str,
    trips_df: str
) -> pd.DataFrame:
    

    stop_times_df["stop_id"]=stop_times_df["stop_id"].astype('str')
    stop_times_df["arrival_sec"] = stop_times_df["arrival_time"].apply(gtfs_time_to_seconds)
    stop_times_df["departure_sec"] = stop_times_df["departure_time"].apply(gtfs_time_to_seconds)

    stop_times_df = stop_times_df.sort_values(["trip_id", "stop_sequence"])

    # consecutive stop pairs
    stop_times_df["next_stop_id"] = stop_times_df.groupby("trip_id")["stop_id"].shift(-1)
    stop_times_df["next_arrival_sec"] = stop_times_df.groupby("trip_id")["arrival_sec"].shift(-1)

    segments = stop_times_df.dropna(subset=["next_stop_id"]).copy()

    # scheduled travel time
    segments["scheduled_travel_time"] = (
        segments["next_arrival_sec"] - segments["departure_sec"]
    )

    min_scheduled = (segments[segments["scheduled_travel_time"]>0]["scheduled_travel_time"].min()) / 2
    segments.loc[
        segments["scheduled_travel_time"] == 0,
        "scheduled_travel_time"
    ] = min_scheduled

    # merge trips
    segments = segments.merge(
        trips_df[
            [
                "trip_id",
                "route_id",
                "direction_id",
                "service_id",
                "shape_id"
            ]
        ],
        on="trip_id",
        how="left"
    )

    # stop coordinates
    stop_coords = stops_df[
        ["stop_id", "stop_lat", "stop_lon"]
    ]

    segments = segments.merge(
        stop_coords.rename(columns={
            "stop_id": "from_stop_id",
            "stop_lat": "from_stop_lat",
            "stop_lon": "from_stop_lon"
        }),
        left_on="stop_id",
        right_on="from_stop_id",
        how="left"
    )

    segments = segments.merge(
        stop_coords.rename(columns={
            "stop_id": "to_stop_id",
            "stop_lat": "to_stop_lat",
            "stop_lon": "to_stop_lon"
        }),
        left_on="next_stop_id",
        right_on="to_stop_id",
        how="left"
    )

    segments["shape_distance"] = haversine( segments["from_stop_lat"], segments["from_stop_lon"], segments["to_stop_lat"], segments["to_stop_lon"] )


    return segments[
        [
            "trip_id",
            "route_id",
            "direction_id",
            "service_id",
            "shape_id",
            "stop_id",
            "next_stop_id",
            "shape_distance",
            "scheduled_travel_time",
            "from_stop_lat",
            "from_stop_lon",
            "to_stop_lat",
            "to_stop_lon",
        ]
    ].rename(columns={
        "stop_id": "from_stop_id",
        "next_stop_id": "to_stop_id"
    })


def load_vehicle_snapshots(snapshot_dir: str) -> pd.DataFrame:
    """
    Load vehicle position snapshots from a directory (GTFS-RT .pb format) and convert to tabular data.
    
    Args:
        snapshot_dir: Path to directory containing GTFS-RT protobuf snapshot files (.pb)
        
    Returns:
        pd.DataFrame with columns: vehicle_id, trip_id, route_id, lat, lon, speed, bearing, timestamp
    """
    # ... existing code ...
    all_vehicles = []
    
    # Process each .pb file in the snapshot directory
    for filename in os.listdir(snapshot_dir):
        if not filename.endswith('vehicle_positions.pb'):
            continue
            
        filepath = os.path.join(snapshot_dir, filename)
        try:
            # Read the protobuf file
            with open(filepath, 'rb') as f:
                pb_data = f.read()
            
            # Parse the protobuf data
            feed = FeedMessage()
            feed.ParseFromString(pb_data)
            
            # Convert protobuf message to dictionary
            feed_dict = MessageToDict(feed)
            
            # Extract vehicles from the feed
            vehicles = feed_dict.get('entity', [])
            
            for vehicle in vehicles:
                # Handle both protobuf and dict formats
                if isinstance(vehicle, dict):
                    vehicle_data = vehicle.get('vehicle', {})
                    trip_data = vehicle_data.get('trip', {})

                
                # Extract required fields
                record = {
                    'vehicle_id': vehicle_data.get('vehicle', {}).get('id', None),
                    'trip_id': trip_data.get('tripId', None),
                    'route_id': trip_data.get('routeId', None),
                    'lat': vehicle_data.get('position', {}).get('latitude', None),
                    'lon': vehicle_data.get('position', {}).get('longitude', None),
                    'speed': vehicle_data.get('position', {}).get('speed', None),
                    'stop_id': vehicle_data.get('stopId', None),
                    'timestamp': vehicle_data.get('timestamp', None)
                }
                
                all_vehicles.append(record)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Convert to DataFrame
            
    df = pd.DataFrame(all_vehicles)
    
    df = pd.DataFrame(df)
    # Ensure timestamp is numeric before conversion
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    return df

def build_trip_duration_map(stop_times: pd.DataFrame):

    st = stop_times.copy()

    st["arrival_secs"] = st["arrival_time"].apply(
        gtfs_time_to_seconds
    )

    trip_duration = (
        st.groupby("trip_id")["arrival_secs"]
        .agg(lambda x: x.max() - x.min())
        .to_dict()
    )

    return trip_duration

def preprocess_vehicle_positions(
    vp: pd.DataFrame,
    stop_times: pd.DataFrame,
    threshold: float = 1.5,
):

    vp = vp.copy()

    required_cols = {"vehicle_id", "trip_id", "timestamp", "lat", "lon"}

    if vp.empty or not required_cols.issubset(vp.columns):
        return vp

    vp = vp.dropna(subset=["timestamp", "lat", "lon"])

    vp["timestamp"] = pd.to_datetime(vp["timestamp"])

    vp = vp.sort_values(
        ["vehicle_id", "trip_id", "timestamp"]
    ).reset_index(drop=True)

    vp["date"] = vp["timestamp"].dt.date

    group_cols = ["vehicle_id", "trip_id", "date"]

    # Scheduled GTFS trip duration (seconds)
    trip_duration_map = build_trip_duration_map(stop_times)

    vp["scheduled_duration_sec"] = (
        vp["trip_id"]
        .map(trip_duration_map)
        .fillna(30 * 60)
    )

    # Maximum allowed duration before splitting
    vp["max_duration_sec"] = (
        vp["scheduled_duration_sec"] * threshold
    )

    # Elapsed time since FIRST point in group
    group_start = (
        vp.groupby(group_cols)["timestamp"]
        .transform("first")
    )

    vp["elapsed_sec"] = (
        vp["timestamp"] - group_start
    ).dt.total_seconds()

    # Number of completed durations
    vp["journey_split"] = np.floor_divide(
        vp["elapsed_sec"],
        vp["max_duration_sec"]
    ).astype(int)

    # Unique journey id
    vp["journey_id"] = (
        vp["vehicle_id"].astype(str)
        + "_"
        + vp["trip_id"].astype(str)
        + "_"
        + vp["date"].astype(str)
        + "_"
        + vp["journey_split"].astype(str)
    )

    vp = vp.drop(columns=["date"])

    print(vp["journey_id"].nunique(), "journeys found")

    return vp


def infer_stop_events(
    vehicle_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    stop_radius_meters: float = 50
):

    stop_events = []

    stop_coords = stops_df[
        ["stop_id", "stop_lat", "stop_lon"]
    ]

    for _, veh in vehicle_df.iterrows():

        dists = haversine(
            veh["lat"],
            veh["lon"],
            stop_coords["stop_lat"],
            stop_coords["stop_lon"]
        )

        min_idx = np.argmin(dists)

        if dists.iloc[min_idx] <= stop_radius_meters:

            stop = stop_coords.iloc[min_idx]

            stop_events.append({
                "vehicle_id": veh["vehicle_id"],
                "trip_id": veh["trip_id"],
                "timestamp": veh["timestamp"],
                "journey_id": veh["journey_id"],
                "stop_id": stop["stop_id"],
                "distance_to_stop": dists.iloc[min_idx]
            })

    return pd.DataFrame(stop_events)

def build_segments_from_events(
    stop_events: pd.DataFrame,
    static_segments: pd.DataFrame
):

    stop_events = stop_events.sort_values(
        ["trip_id", "timestamp"]
    )

    segments = []

    for journey_id, grp in stop_events.groupby("journey_id"):

        grp = grp.sort_values("timestamp")

        rows = grp.to_dict("records")

        for i in range(len(rows) - 1):

            curr = rows[i]
            nxt = rows[i + 1]

            # VALIDATE CONSECUTIVE STOPS
            static_match = static_segments[
                (static_segments["trip_id"] == curr["trip_id"])
                &
                (static_segments["from_stop_id"] == curr["stop_id"])
                &
                (static_segments["to_stop_id"] == nxt["stop_id"])
            ]

            if static_match.empty:
                continue

            travel_time = (
                nxt["timestamp"] - curr["timestamp"]
            ).total_seconds()

            if travel_time <= 0:
                continue

            static_row = static_match.iloc[0]

            segments.append({

                "trip_id": curr["trip_id"],
                "journey_id": journey_id,
                "from_stop_id": curr["stop_id"],
                "to_stop_id": nxt["stop_id"],
                "departure_time": curr["timestamp"],
                "arrival_time": nxt["timestamp"],
                "actual_travel_time": travel_time,
                "scheduled_travel_time":
                    static_row["scheduled_travel_time"],
                "delay":
                    travel_time
                    - static_row["scheduled_travel_time"],
                "shape_distance":
                    static_row["shape_distance"],
                "route_id":
                    static_row["route_id"],
                "direction_id":
                    static_row["direction_id"],
                "service_id":
                    static_row["service_id"]
            })

    return pd.DataFrame(segments)


def features_engineering(segments: pd.DataFrame):

    df = segments.copy()

    df["hour"] = df["departure_time"].dt.hour

    df["weekday"] = df["departure_time"].dt.weekday

    df["is_peak"] = (
        (
            (df["hour"] >= 7)
            & (df["hour"] <= 9)
        )
        |
        (
            (df["hour"] >= 16)
            & (df["hour"] <= 19)
        )
    ).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df = df.sort_values([
        "trip_id",
        "journey_id",
        "departure_time"   # or stop_sequence if available (better)
        ])

    df["previous_delay"] = (
        df.groupby(["trip_id", "journey_id"])["delay"]
        .shift(1)
        .fillna(0)
    )


    df["previous_delay_ratio"] = df["previous_delay"] / (df["scheduled_travel_time"] + 1e-6)

    df["previous_delay_delta"] = (
        df["previous_delay"]
        - df["previous_delay"].shift(1).fillna(0)
    )

    return df

def save_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_dir: str,
    dataset_names: List[str] = None,
    timestamp: bool = True,
    **kwargs
) -> None:
    """
    Save datasets with optional train/test split.
    
    Adds:
    - train/test split per dataset
    - separate saved files for each split
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_format = kwargs.get('save_format', 'csv')
    test_size = kwargs.get('test_size', 0.2)
    random_state = kwargs.get('random_state', 42)
    split = kwargs.get('split', True)  # enable/disable splitting

    if dataset_names is None:
        dataset_names = list(datasets.keys())

    for name, df in datasets.items():
        if name not in dataset_names:
            continue

        if df.empty:
            print(f"Skipping empty dataset: {name}")
            continue

        # =========================
        # TRAIN / TEST SPLIT
        # =========================

        gss = GroupShuffleSplit(
            test_size=test_size,
            random_state=random_state
        )

        train_idx, test_idx = next(
            gss.split(df, groups=df["journey_id"])
        )

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        split_dfs = {
            "train": train_df,
            "test": test_df
        }
        # =========================
        # SAVE FILES
        # =========================
        for split_name, split_df in split_dfs.items():

            if split_df.empty:
                continue

            ext = "parquet" if save_format == 'parquet' else "csv"

            filename = f"{name}_{split_name}"

            if timestamp:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename += f"_{timestamp_str}"

            filename += f".{ext}"

            filepath = os.path.join(output_dir, filename)

            try:
                if save_format == 'parquet':
                    split_df.to_parquet(filepath, index=False)
                else:
                    split_df.to_csv(filepath, index=False, sep=kwargs.get('separator', ','))

                print(f"Saved {name} ({split_name}) → {filepath} ({len(split_df)} rows)")

            except Exception as e:
                print(f"Error saving {name} ({split_name}): {e}")

def load_trip_updates(snapshot_dir: str) -> pd.DataFrame:
    """
    Load trip update snapshots from a directory (GTFS-RT .pb format) and convert to tabular data.
    
    Args:
        snapshot_dir: Path to directory containing GTFS-RT protobuf snapshot files (.pb)
        
    Returns:
        pd.DataFrame with columns: trip_id, stop_id, arrival_delay, departure_delay,
                                   arrival_time, timestamp
    """
    # ... existing code ...
    all_trip_updates = []
    
    # Process each .pb file in the snapshot directory
    for filename in os.listdir(snapshot_dir):
        if not filename.endswith('.pb'):
            continue
            
        filepath = os.path.join(snapshot_dir, filename)
        try:
            # Read the protobuf file
            with open(filepath, 'rb') as f:
                pb_data = f.read()
            
            # Parse the protobuf data
            feed = FeedMessage()
            feed.ParseFromString(pb_data)
            
            # Convert protobuf message to dictionary
            feed_dict = MessageToDict(feed)
            
            # Extract entities from the feed
            entities = feed_dict.get('entity', [])
            
            for entity in entities:
                # Process only trip update entities
                if 'tripUpdate' not in entity:
                    continue
                    
                trip_update = entity.get('tripUpdate', {})
                # Get trip information
                trip_info = trip_update.get('trip', {})
                trip_id = trip_info.get('tripId', None)
                route_id = trip_info.get('routeId', None)
                start_time = trip_info.get('startTime', None)
                start_date = trip_info.get('startDate',None)
                # Get update timestamp
                timestamp = trip_update.get('timestamp', None)
                
                # Process stop time updates
                stop_time_updates = trip_update.get('stopTimeUpdate', [])
                
                if not stop_time_updates:
                    # Handle case where there are no specific stop time updates
                    record = {
                        'trip_id': trip_id,
                        'route_id': route_id,
                        'start_time': start_time,
                        'start_date':start_date,
                        'stop_id': None,
                        'arrival_delay': None,
                        'departure_delay': None,
                        'arrival_time': None,
                        'timestamp': timestamp
                    }
                    all_trip_updates.append(record)
                else:
                    # Extract information for each stop time update
                    for stop_time in stop_time_updates:
                        record = {
                            'trip_id': trip_id,
                            'route_id': route_id,
                            'start_time': start_time,
                            'start_date':start_date,
                            'stop_id': stop_time.get('stopId', None),
                            'vehicle_id': trip_update.get('vehicle', {}).get('id', None),
                            'arrival_delay': stop_time.get('arrival', {}).get('delay', None),
                            'departure_delay': stop_time.get('departure', {}).get('delay', None),
                            'stop_sequence': stop_time.get('stopSequence',None),
                            'timestamp': timestamp
                        }
                        all_trip_updates.append(record)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trip_updates)
    # Ensure timestamp is numeric before conversion
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    return df


def main(
    vehicle_snapshot_dir = "./data/mobility-datasets/GTFS_RT/",
    gtfs_path  = "./data/mobility-datasets/GTFS_Scheduled/",
    output_dir: str ="./data/mobility-datasets/"
) -> pd.DataFrame:
    """
    Main function to orchestrate the GTFS-RT data processing pipeline.
    
    Args:
        vehicle_snapshot_dir: Directory containing vehicle position .pb snapshots
        trip_update_snapshot_dir: Directory containing trip update .pb snapshots
        gtfs_path: Optional path to GTFS directory
        
    Returns:
        pd.DataFrame with joined vehicle positions, trip updates, and scheduled GTFS info
    """

    datasets = {}
    stops_times_path = os.path.join(gtfs_path,"stop_times.txt")
    stops_path = os.path.join(gtfs_path,"stops.txt")
    trips_path = os.path.join(gtfs_path,"trips.txt")

    #Load GTFS static files

    stop_times_df = pd.read_csv(stops_times_path)
    trips_df = pd.read_csv(trips_path)
    stops_df = pd.read_csv(stops_path)

    #build static segments
    print(f"Building static segments informations...")
    static_segments_df=build_static_segments(stop_times_df,stops_df,trips_df)
    print(f"Builded {len(static_segments_df)} static segments")


    # Load vehicle snapshots
    print(f"Loading vehicle snapshots from {vehicle_snapshot_dir} and preprocessing them")
    vehicle_df = load_vehicle_snapshots(vehicle_snapshot_dir)
    vehicle_df=preprocess_vehicle_positions(vehicle_df,stop_times_df)
    print(f"Loaded and preprocessed  {len(vehicle_df)} vehicle records")

    
    #Infering stop events
    print(f"Infering stop events")
    stop_events_df = infer_stop_events(vehicle_df,stops_df)

    #build_realtime segments
    print(f"Building realtime segments from stop events infered and static segments")
    segments_df = build_segments_from_events(stop_events_df,static_segments_df)

    print(f"Feature engineering")
    segments_df = features_engineering(segments_df)

    datasets['segments'] = segments_df

    if output_dir:
        print(f"Saving datasets to {output_dir}")
        
        # Save the datasets
        save_datasets(
            datasets=datasets,
            output_dir=output_dir,
            dataset_names=list(datasets.keys()),
            timestamp=False,
            save_format='csv'  # or 'parquet'
        )
    
if __name__ == "__main__":
    main()
