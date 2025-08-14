# api/generator.py
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import os

# Path to your real filtered dataset (used only to learn structure)
REAL_DATA_PATH = os.getenv("REAL_DATA_PATH", "/data/startups_filtered.csv")

def generate_single_synthetic_startup():
    """Generate exactly one synthetic startup using SDV"""
    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(f"Real data not found at {REAL_DATA_PATH}")

    print(f"Loading real data from {REAL_DATA_PATH}")
    data = pd.read_csv(REAL_DATA_PATH)

    if data.empty:
        raise ValueError("Real data is empty")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    synthesizer = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
    synthesizer.fit(data)

    synthetic_data = synthesizer.sample(1)  # Only one startup

    # Convert to dict (flatten if needed)
    result = synthetic_data.iloc[0].to_dict()

    # Ensure all values are JSON-serializable
    for k, v in result.items():
        if pd.isna(v):
            result[k] = None
        elif isinstance(v, (pd.Int64Dtype, int)):
            result[k] = int(v) if not pd.isna(v) else None
        elif isinstance(v, float):
            result[k] = float(v) if not pd.isna(v) else None
        else:
            result[k] = str(v)

    return result
