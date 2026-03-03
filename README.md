# 📡 6G Network Slicing — Time-Series Enrichment & Dashboard

A two-part Python project that enriches a synthetic 6G network slicing dataset with realistic time-series resource signals and visualises them in an interactive Streamlit dashboard.

---

## Project Structure

```
Simulation-forecasting/
├── network_slicing_dataset - v3.csv          # Original dataset (input)
├── enrich_6g.py                              # Part 1 — Dataset enrichment
├── app.py                                    # Part 2 — Streamlit dashboard
├── requirements.txt                          # Python dependencies
├── README.md
└── data/
    └── network_slicing_dataset_enriched_timeseries.csv   # Generated output
```

---

## Quick Start

### 1. Create & activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Enrich the dataset

```bash
python enrich_6g.py
```

This reads `network_slicing_dataset - v3.csv` and writes the enriched file to `data/`.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Part 1 — Dataset Enrichment (`enrich_6g.py`)

### What it does

Reads the raw 6G CSV (10,000 rows, 13 columns) and generates a rich time-series dataset (21,600 rows, 21 columns) by:

- Creating **3 slice instances per slice type** (15 `slice_id` values total)
- Adding a **`timestamp` column** at 1-minute intervals (24 h per instance)
- Generating **5 realistic resource signal columns** per instance
- Deriving a **binary `congestion_flag`**

### Signal model

Each signal is built from three components layered together:

| Component | Description |
|---|---|
| **Daily curve** | Sinusoidal shape — trough at midnight, peak around noon |
| **Random bursts** | Sporadic load spikes with exponential decay |
| **Gaussian noise** | Per-minute random variation |

Cross-signal correlation chain:

```
active_users  →  bw_util_pct  →  cpu_util_pct
```

`mem_util_pct` and `queue_len` are semi-independent (slow drift + CPU excess).

### Congestion rule

```
congestion_flag = 1  if  cpu_util_pct > 85
                      OR  bw_util_pct  > 90
                      OR  queue_len    > 20
```

### Configuration (top of `enrich_6g.py`)

| Constant | Default | Meaning |
|---|---|---|
| `INSTANCES_PER_SLICE_TYPE` | 3 | Slice instances per type |
| `MINUTES_PER_INSTANCE` | 1440 | Minutes of data per instance (1 day) |
| `START_TIME` | 2025-01-01 00:00 | First timestamp |
| `RANDOM_SEED` | 42 | Reproducibility seed |

---

## Part 2 — Streamlit Dashboard (`app.py`)

### Features

**Sidebar controls**
- Select `slice_id` (dropdown)
- Select time window (start / end sliders)

**Summary KPI row** (6 metric cards)
- ⚙️ Avg CPU %
- 📶 Avg Bandwidth %
- 🧠 Avg Memory %
- 👥 Avg Active Users
- 🔴 % Time Congested
- 📦 Max Queue Length

**Resource utilisation charts** (Plotly, interactive)
- CPU utilisation over time — with 85% threshold line
- Bandwidth utilisation over time — with 90% threshold line
- Memory utilisation over time
- Active users over time

**Congestion section**
- Congestion flag timeline (Clear / Congested)
- Queue length over time — with queue > 20 threshold line

**Expandable panels**
- 🗄️ Raw data table (first 200 rows)
- 📐 Distribution histograms (CPU, BW, Memory)

---

## Dataset Column Reference

### Columns kept from the original CSV

| Column | Description |
|---|---|
| `Use Case Type` | Application scenario (e.g. VR/AR, V2X, Smart City…) |
| `Packet Loss Budget` | SLA packet-loss tolerance |
| `Latency Budget (ns)` | SLA latency target |
| `Jitter Budget (ns)` | SLA jitter target |
| `Data Rate Budget (Gbps)` | SLA data-rate requirement |
| `Required Mobility` | Mobility class |
| `Required Connectivity` | Connectivity type |
| `Slice Available Transfer Rate (Gbps)` | Allocated throughput |
| `Slice Latency (ns)` | Measured slice latency |
| `Slice Packet Loss` | Measured packet loss |
| `Slice Jitter (ns)` | Measured jitter |
| `Slice Type` | `ERLLC` / `umMTC` / `MBRLLC` / `mURLLC` / `feMBB` |
| `Slice Handover` | Handover event flag |

### Columns added by enrichment

| Column | Type | Range | Description |
|---|---|---|---|
| `timestamp` | datetime | 2025-01-01 → 2025-01-03 | 1-minute resolution timestamp |
| `slice_type` | string | 5 types | Copy of `Slice Type` for convenience |
| `slice_id` | string | 15 IDs | `{type}_{01–03}` instance identifier |
| `cpu_util_pct` | float | 30 – 99.9 % | CPU utilisation (correlated with BW) |
| `mem_util_pct` | float | 28 – 83 % | Memory utilisation (semi-independent) |
| `bw_util_pct` | float | 23 – 99.9 % | Bandwidth utilisation (correlated with users) |
| `active_users` | int | 30 – 458 | Active user count |
| `queue_len` | int | 0 – 26 | Packet queue length |
| `congestion_flag` | int (0/1) | — | 1 = congested, 0 = clear |

### Enriched dataset statistics

| Metric | Value |
|---|---|
| Total rows | 21,600 |
| Total columns | 21 |
| Slice IDs | ERLLC_01–03, MBRLLC_01–03, feMBB_01–03, mURLLC_01–03, umMTC_01–03 |
| Time range | 2025-01-01 00:00 → 2025-01-03 23:59 |
| Avg CPU utilisation | 69.7 % |
| Avg BW utilisation | 72.5 % |
| Avg Memory utilisation | 56.0 % |
| Avg active users | 138 |
| % time congested | **23.4 %** |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading & manipulation |
| `numpy` | ≥ 1.26 | Signal generation |
| `streamlit` | ≥ 1.35 | Dashboard server |
| `plotly` | ≥ 5.22 | Interactive charts |

```bash
pip install -r requirements.txt
```

---

## Slice Type Profiles

Each slice type has its own resource baseline, shaping how signals behave:

| Slice Type | CPU base | Mem base | BW base | User base | Typical use |
|---|---|---|---|---|---|
| `ERLLC` | 55 % | 50 % | 60 % | 80 | Ultra-reliable low-latency |
| `umMTC` | 35 % | 40 % | 30 % | 200 | Massive IoT / machine-type |
| `MBRLLC` | 70 % | 65 % | 75 % | 60 | Enhanced reliable broadband |
| `mURLLC` | 60 % | 55 % | 65 % | 50 | Modified URLLC |
| `feMBB` | 80 % | 70 % | 85 % | 300 | Enhanced mobile broadband |
