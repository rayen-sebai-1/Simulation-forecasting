"""
enrich_6g.py
------------
Enriches the raw 6G network slicing dataset with realistic time-series
resource utilisation columns:
    - timestamp (1-minute interval)
    - slice_id  (multiple instances per slice type)
    - cpu_util_pct
    - mem_util_pct
    - bw_util_pct
    - active_users
    - queue_len
    - congestion_flag  (binary)

Signal model
    base = sinusoidal daily load curve
    + random burst events
    + gaussian noise
    + cross-signal correlation (users <-> BW <-> CPU)

Output: data/network_slicing_dataset_enriched_timeseries.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

RAW_CSV = Path("network_slicing_dataset - v3.csv")
OUT_DIR = Path("data")
OUT_CSV = OUT_DIR / "network_slicing_dataset_enriched_timeseries.csv"

START_TIME = pd.Timestamp("2025-01-01 00:00:00")
INSTANCES_PER_SLICE_TYPE = 3          # slice_id instances per type
MINUTES_PER_INSTANCE = 1440           # 24 h × 60 min = 1 day per instance

# Per slice-type resource baselines (realistic 6G defaults)
SLICE_PROFILES: dict[str, dict] = {
    "ERLLC":  {"cpu_base": 55, "mem_base": 50, "bw_base": 60,  "user_base": 80,  "user_scale": 40},
    "umMTC":  {"cpu_base": 35, "mem_base": 40, "bw_base": 30,  "user_base": 200, "user_scale": 150},
    "MBRLLC": {"cpu_base": 70, "mem_base": 65, "bw_base": 75,  "user_base": 60,  "user_scale": 35},
    "mURLLC": {"cpu_base": 60, "mem_base": 55, "bw_base": 65,  "user_base": 50,  "user_scale": 30},
    "feMBB":  {"cpu_base": 80, "mem_base": 70, "bw_base": 85,  "user_base": 300, "user_scale": 200},
}

RANDOM_SEED = 42

# ── Helper functions ──────────────────────────────────────────────────────────

def daily_curve(n: int, amplitude: float = 1.0, phase_shift: float = 0.0) -> np.ndarray:
    """Return a smooth sinusoidal daily load curve (peak at noon).

    Args:
        n: number of time steps (minutes)
        amplitude: peak-to-trough half-amplitude
        phase_shift: phase offset in radians
    """
    t = np.arange(n)
    # Two full cycles over `n` minutes so we can see the pattern even for n<1440
    freq = 2 * np.pi / 1440
    return amplitude * np.sin(freq * t - np.pi / 2 + phase_shift)  # trough at midnight, peak at noon


def random_bursts(n: int, rng: np.random.Generator,
                  burst_prob: float = 0.002,
                  burst_scale: float = 15.0,
                  decay: float = 0.92) -> np.ndarray:
    """Inject sporadic load spikes that decay exponentially.

    Args:
        n: length of the series
        rng: numpy random Generator
        burst_prob: probability of a burst starting at any given minute
        burst_scale: peak height of a burst
        decay: exponential decay factor per minute
    """
    signal = np.zeros(n)
    current = 0.0
    for i in range(n):
        if rng.random() < burst_prob:
            current += rng.uniform(burst_scale * 0.5, burst_scale)
        signal[i] = current
        current *= decay
    return signal


def gaussian_noise(n: int, rng: np.random.Generator, std: float = 2.5) -> np.ndarray:
    return rng.normal(0, std, n)


def clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_timeseries_for_instance(
    slice_type: str,
    instance_idx: int,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one day of time-series data for a single slice instance.

    The correlation chain is:
        active_users  →  bw_util_pct  →  cpu_util_pct
    with independent noise and bursts added at each stage.
    """
    profile = SLICE_PROFILES[slice_type]
    phase = instance_idx * 0.4   # slight phase offset between instances

    # ── active_users ──────────────────────────────────────────────────────────
    user_curve = daily_curve(n, amplitude=profile["user_scale"] * 0.5, phase_shift=phase)
    user_bursts = random_bursts(n, rng, burst_prob=0.003, burst_scale=profile["user_scale"] * 0.3)
    user_noise = gaussian_noise(n, rng, std=profile["user_scale"] * 0.05)
    active_users = profile["user_base"] + user_curve + user_bursts + user_noise
    active_users = clip(active_users, 1, profile["user_base"] + profile["user_scale"] * 1.2).astype(int)

    # ── bw_util_pct (correlated with users) ───────────────────────────────────
    user_norm = (active_users - active_users.min()) / (active_users.max() - active_users.min() + 1e-6)
    bw_base_signal = profile["bw_base"] + user_norm * 20
    bw_bursts = random_bursts(n, rng, burst_prob=0.002, burst_scale=12.0)
    bw_noise = gaussian_noise(n, rng, std=3.0)
    bw_util_pct = bw_base_signal + bw_bursts + bw_noise
    bw_util_pct = clip(bw_util_pct, 5.0, 99.9)

    # ── cpu_util_pct (correlated with BW) ─────────────────────────────────────
    bw_norm = (bw_util_pct - bw_util_pct.min()) / (bw_util_pct.max() - bw_util_pct.min() + 1e-6)
    cpu_base_signal = profile["cpu_base"] + bw_norm * 18
    cpu_bursts = random_bursts(n, rng, burst_prob=0.0025, burst_scale=10.0)
    cpu_noise = gaussian_noise(n, rng, std=2.5)
    cpu_util_pct = cpu_base_signal + cpu_bursts + cpu_noise
    cpu_util_pct = clip(cpu_util_pct, 5.0, 99.9)

    # ── mem_util_pct (semi-independent, slow drift) ───────────────────────────
    mem_curve = daily_curve(n, amplitude=8.0, phase_shift=phase + 0.8)
    mem_noise = gaussian_noise(n, rng, std=1.5)
    mem_util_pct = profile["mem_base"] + mem_curve + mem_noise
    mem_util_pct = clip(mem_util_pct, 10.0, 99.9)

    # ── queue_len (correlated with CPU congestion) ────────────────────────────
    cpu_excess = np.maximum(cpu_util_pct - 75, 0)
    queue_len = (cpu_excess * 0.8 + random_bursts(n, rng, burst_prob=0.003, burst_scale=8.0)
                 + gaussian_noise(n, rng, std=1.0))
    queue_len = clip(queue_len, 0, 50).astype(int)

    # ── congestion_flag ───────────────────────────────────────────────────────
    congestion_flag = (
        (cpu_util_pct > 85) |
        (bw_util_pct > 90) |
        (queue_len > 20)
    ).astype(int)

    # ── timestamps ────────────────────────────────────────────────────────────
    global_offset_min = instance_idx * n
    timestamps = pd.date_range(
        start=START_TIME + pd.Timedelta(minutes=global_offset_min),
        periods=n,
        freq="1min",
    )

    return pd.DataFrame({
        "timestamp":      timestamps,
        "slice_type":     slice_type,
        "slice_id":       f"{slice_type}_{instance_idx + 1:02d}",
        "cpu_util_pct":   np.round(cpu_util_pct, 2),
        "mem_util_pct":   np.round(mem_util_pct, 2),
        "bw_util_pct":    np.round(bw_util_pct, 2),
        "active_users":   active_users,
        "queue_len":      queue_len,
        "congestion_flag": congestion_flag,
    })


# ── Load raw CSV ──────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw 6G network slicing CSV (semicolon-delimited)."""
    # Try semicolon first, fall back to comma
    df = pd.read_csv(path, sep=";")
    if df.shape[1] < 2:
        df = pd.read_csv(path)
    print(f"[load_raw] Loaded {len(df):,} rows × {df.shape[1]} cols from '{path}'")
    return df


# ── Main enrichment pipeline ──────────────────────────────────────────────────

def enrich(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build enriched time-series DataFrame from the raw dataset."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Detect which slice types are present in the raw data
    slice_col = next(
        (c for c in raw_df.columns if "type" in c.lower() and "slice" in c.lower()),
        None,
    )
    if slice_col and raw_df[slice_col].nunique() > 0:
        detected_types = raw_df[slice_col].dropna().unique().tolist()
        # Keep only types we have profiles for
        slice_types = [s for s in detected_types if s in SLICE_PROFILES]
        if not slice_types:
            slice_types = list(SLICE_PROFILES.keys())
    else:
        slice_types = list(SLICE_PROFILES.keys())

    print(f"[enrich] Slice types: {slice_types}")

    # Keep extra context columns from the raw dataset
    extra_cols = [c for c in raw_df.columns if c != slice_col]
    # Map slice type → sampled row of raw attributes
    raw_by_type: dict[str, pd.DataFrame] = {}
    if slice_col:
        for st in slice_types:
            mask = raw_df[slice_col] == st
            raw_by_type[st] = raw_df[mask].reset_index(drop=True)

    records: list[pd.DataFrame] = []

    for st in slice_types:
        for inst in range(INSTANCES_PER_SLICE_TYPE):
            n = MINUTES_PER_INSTANCE
            ts_df = generate_timeseries_for_instance(st, inst, n, rng)

            # Attach a random raw-data row as contextual metadata
            if st in raw_by_type and len(raw_by_type[st]) > 0:
                sample_row = raw_by_type[st].sample(n=1, random_state=RANDOM_SEED + inst).iloc[0]
                for col in extra_cols:
                    if col not in ts_df.columns:
                        ts_df[col] = sample_row.get(col, np.nan)

            records.append(ts_df)
            print(
                f"  [enrich] {st} instance {inst + 1}/{INSTANCES_PER_SLICE_TYPE} "
                f"— {n} rows generated"
            )

    enriched = pd.concat(records, ignore_index=True)
    enriched.sort_values(["slice_id", "timestamp"], inplace=True)
    enriched.reset_index(drop=True, inplace=True)
    print(f"[enrich] Enriched dataset: {len(enriched):,} rows × {enriched.shape[1]} cols")
    return enriched


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw(RAW_CSV)
    enriched_df = enrich(raw_df)

    enriched_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved enriched dataset → {OUT_CSV}")
    print(f"   Rows: {len(enriched_df):,}  |  Columns: {enriched_df.shape[1]}")
    print(f"   Slice IDs : {sorted(enriched_df['slice_id'].unique())}")
    print(f"   Time range: {enriched_df['timestamp'].min()} → {enriched_df['timestamp'].max()}")
    congestion_rate = enriched_df["congestion_flag"].mean() * 100
    print(f"   Congestion : {congestion_rate:.1f}% of all records")


if __name__ == "__main__":
    main()
