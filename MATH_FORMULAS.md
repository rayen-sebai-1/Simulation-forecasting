# Mathematical Formulas & Methods — 6G Network Slicing Simulation

> **Project:** Simulation-forecasting / 6G Network Slicing Dataset Enrichment  
> **Files documented:** `enrich_6g.py`, `app.py`

---

## Table of Contents

1. [Signal Architecture Overview](#1-signal-architecture-overview)
2. [Daily Load Curve — Sinusoidal Model](#2-daily-load-curve--sinusoidal-model)
3. [Random Burst Events — Exponential Decay Model](#3-random-burst-events--exponential-decay-model)
4. [Gaussian Noise](#4-gaussian-noise)
5. [Phase Offset (Per-Instance Differentiation)](#5-phase-offset-per-instance-differentiation)
6. [Active Users Signal Composition](#6-active-users-signal-composition)
7. [Min-Max Normalization (Cross-Signal Correlation Driver)](#7-min-max-normalization-cross-signal-correlation-driver)
8. [Bandwidth Utilization — Correlated with Users](#8-bandwidth-utilization--correlated-with-users)
9. [CPU Utilization — Correlated with Bandwidth](#9-cpu-utilization--correlated-with-bandwidth)
10. [Memory Utilization — Semi-Independent Slow Drift](#10-memory-utilization--semi-independent-slow-drift)
11. [Queue Length — Derived from CPU Excess](#11-queue-length--derived-from-cpu-excess)
12. [Congestion Flag — Multi-Condition Binary Classifier](#12-congestion-flag--multi-condition-binary-classifier)
13. [Clipping (Signal Bounding)](#13-clipping-signal-bounding)
14. [Timestamp Generation](#14-timestamp-generation)
15. [Summary Metrics (Dashboard — app.py)](#15-summary-metrics-dashboard--apppy)
16. [Signal Correlation Chain Diagram](#16-signal-correlation-chain-diagram)

---

## 1. Signal Architecture Overview

Each generated time-series signal is a **superposition** of three independent components:

$$S(t) = S_{\text{base}}(t) + S_{\text{burst}}(t) + \eta(t)$$

Where:

| Symbol | Meaning |
|---|---|
| $S_{\text{base}}(t)$ | Deterministic sinusoidal daily load curve |
| $S_{\text{burst}}(t)$ | Stochastic exponentially-decaying burst events |
| $\eta(t)$ | Additive Gaussian white noise |

This architecture is applied independently to each resource metric (CPU, memory, bandwidth, active users), with **cross-signal correlation** introduced via normalized intermediate signals (see §7).

---

## 2. Daily Load Curve — Sinusoidal Model

**Origin:** Classic circadian / diurnal traffic model used in network simulation (3GPP TR 36.814, ITU-T traffic models).

### Formula

$$\text{daily\_curve}(t) = A \cdot \sin\!\left(\frac{2\pi}{T} \cdot t - \frac{\pi}{2} + \varphi\right)$$

### Parameters

| Symbol | Variable | Value | Meaning |
|---|---|---|---|
| $A$ | `amplitude` | signal-dependent | Peak-to-trough half-amplitude |
| $T$ | — | $1440$ min | Period = one full day (24 h × 60 min) |
| $t$ | `t = np.arange(n)` | $0, 1, \ldots, n-1$ | Time index in minutes |
| $\varphi$ | `phase_shift` | slice/instance-dependent | Phase offset in radians |
| $-\pi/2$ | — | constant | Shifts the sine so the **trough is at $t=0$ (midnight)** and **peak at $t=720$ (noon)** |

### Angular Frequency

$$\omega = \frac{2\pi}{T} = \frac{2\pi}{1440} \approx 0.004363\ \text{rad/min}$$

### Behavior

- At $t = 0$ (midnight): $\sin(-\pi/2) = -1 \Rightarrow$ minimum load
- At $t = 720$ (noon): $\sin(\pi/2) = +1 \Rightarrow$ maximum load
- Full oscillation restores at $t = 1440$ (next midnight)

### Code Reference (`enrich_6g.py`, lines 53–64)

```python
freq = 2 * np.pi / 1440
return amplitude * np.sin(freq * t - np.pi / 2 + phase_shift)
```

---

## 3. Random Burst Events — Exponential Decay Model

**Origin:** Poisson-arrival + exponential-decay model widely used in queueing theory and network traffic engineering (see: Kleinrock, *Queueing Systems*, 1975; also used in 5G/6G burst traffic simulation).

### Burst Arrival Process

Bursts arrive according to a **Bernoulli process** (discrete-time approximation of a Poisson process):

$$P(\text{burst at time } t) = p_b$$

Where $p_b$ is `burst_prob` (typically $0.002$ to $0.003$ per minute).

### Burst Magnitude

When a burst is triggered at time $\tau$, its magnitude is drawn uniformly:

$$M_\tau \sim \mathcal{U}\!\left[\frac{b}{2},\ b\right]$$

Where $b$ is `burst_scale`.

### Exponential Decay

After triggering, the burst decays geometrically (discrete-time exponential decay):

$$S_{\text{burst}}(t) = \sum_{\tau \leq t} M_\tau \cdot d^{\,t - \tau}$$

Where $d$ is `decay` (typically $0.92$), so the **half-life** of a burst is:

$$t_{1/2} = \frac{\ln 2}{-\ln d} = \frac{0.6931}{-\ln(0.92)} \approx \frac{0.6931}{0.0834} \approx 8.3\ \text{minutes}$$

### Iterative Implementation

The recurrence implemented in the code is equivalent to the above:

$$\text{current}_{t} = \text{current}_{t-1} \cdot d + M_t \cdot \mathbf{1}[\text{burst}_{t}]$$

$$S_{\text{burst}}(t) = \text{current}_t$$

### Code Reference (`enrich_6g.py`, lines 67–87)

```python
current = 0.0
for i in range(n):
    if rng.random() < burst_prob:
        current += rng.uniform(burst_scale * 0.5, burst_scale)
    signal[i] = current
    current *= decay
```

### Burst Parameters Used per Signal

| Signal | `burst_prob` | `burst_scale` |
|---|---|---|
| `active_users` | $0.003$ | `user_scale × 0.3` |
| `bw_util_pct` | $0.002$ | $12.0$ |
| `cpu_util_pct` | $0.0025$ | $10.0$ |
| `queue_len` | $0.003$ | $8.0$ |

---

## 4. Gaussian Noise

**Origin:** Additive White Gaussian Noise (AWGN) — standard model for measurement and environmental uncertainty across all time-series simulation literature.

### Formula

$$\eta(t) \sim \mathcal{N}(0,\ \sigma^2)$$

Implemented as:

$$\eta(t) = \sigma \cdot Z, \quad Z \sim \mathcal{N}(0, 1)$$

### Parameters per Signal

| Signal | $\sigma$ (`std`) |
|---|---|
| `active_users` | `user_scale × 0.05` |
| `bw_util_pct` | $3.0$ |
| `cpu_util_pct` | $2.5$ |
| `mem_util_pct` | $1.5$ |
| `queue_len` | $1.0$ |

### Code Reference (`enrich_6g.py`, line 91)

```python
return rng.normal(0, std, n)
```

---

## 5. Phase Offset (Per-Instance Differentiation)

**Origin:** Common technique in network simulation to prevent all slice instances from being perfectly synchronized, mimicking real-world traffic diversity.

### Formula

$$\varphi_i = i \cdot \Delta\varphi$$

Where:

| Symbol | Value | Meaning |
|---|---|---|
| $i$ | `instance_idx` ∈ {0, 1, 2} | Instance index |
| $\Delta\varphi$ | $0.4$ rad | Phase increment between instances |

### Code Reference (`enrich_6g.py`, line 111)

```python
phase = instance_idx * 0.4
```

**Effect:** Instance 0 starts at phase $0$, instance 1 at $0.4$ rad (~23 min ahead), instance 2 at $0.8$ rad (~46 min ahead), creating visibly distinct but similarly-shaped daily profiles.

---

## 6. Active Users Signal Composition

**Origin:** Composite signal model = deterministic trend + stochastic events + noise (Box-Jenkins ARIMA family; also standard in 3GPP traffic simulation).

### Full Formula

$$U(t) = U_{\text{base}} + A_u \cdot \sin\!\left(\omega t - \frac{\pi}{2} + \varphi\right) + S_{\text{burst},u}(t) + \eta_u(t)$$

Where:

| Symbol | Code variable | Value |
|---|---|---|
| $U_{\text{base}}$ | `user_base` | slice-type dependent (see §A) |
| $A_u$ | `user_scale × 0.5` | half of user scale range |
| $\omega$ | $2\pi / 1440$ | daily frequency |
| $\varphi$ | `phase` | instance phase offset |
| $S_{\text{burst},u}$ | `user_bursts` | burst model with $p_b = 0.003$ |
| $\eta_u$ | `user_noise` | $\mathcal{N}(0,\ (0.05 \cdot \text{user\_scale})^2)$ |

### Final value (clipped, integer)

$$U(t) = \left\lfloor \text{clip}\!\left(U(t),\ 1,\ U_{\text{base}} + 1.2 \cdot \text{user\_scale}\right) \right\rfloor$$

### Code Reference (`enrich_6g.py`, lines 113–118)

```python
user_curve = daily_curve(n, amplitude=profile["user_scale"] * 0.5, phase_shift=phase)
user_bursts = random_bursts(n, rng, burst_prob=0.003, burst_scale=profile["user_scale"] * 0.3)
user_noise = gaussian_noise(n, rng, std=profile["user_scale"] * 0.05)
active_users = profile["user_base"] + user_curve + user_bursts + user_noise
active_users = clip(active_users, 1, profile["user_base"] + profile["user_scale"] * 1.2).astype(int)
```

---

## 7. Min-Max Normalization (Cross-Signal Correlation Driver)

**Origin:** Feature scaling / normalization — standard technique from statistics and machine learning used here to couple signal magnitudes across different metrics.

### Formula

$$\hat{X}(t) = \frac{X(t) - \min(X)}{\max(X) - \min(X) + \varepsilon}$$

Where $\varepsilon = 10^{-6}$ is a small constant to prevent division by zero.

This normalized signal $\hat{X}(t) \in [0, 1]$ is then used to **drive** a downstream signal, creating statistical correlation.

### Code Reference (`enrich_6g.py`, lines 121, 129)

```python
# Users → BW correlation
user_norm = (active_users - active_users.min()) / (active_users.max() - active_users.min() + 1e-6)

# BW → CPU correlation
bw_norm = (bw_util_pct - bw_util_pct.min()) / (bw_util_pct.max() - bw_util_pct.min() + 1e-6)
```

---

## 8. Bandwidth Utilization — Correlated with Users

**Origin:** Empirically motivated: more active users → higher bandwidth demand. Linear coupling via normalized user signal.

### Formula

$$\text{BW}(t) = B_{\text{bw}} + 20 \cdot \hat{U}(t) + S_{\text{burst,bw}}(t) + \eta_{\text{bw}}(t)$$

Where:

| Symbol | Code variable | Value |
|---|---|---|
| $B_{\text{bw}}$ | `bw_base` | slice-type dependent |
| $\hat{U}(t)$ | `user_norm` | min-max normalized active users |
| $20$ | — | coupling weight (% BW range added by users) |
| $S_{\text{burst,bw}}$ | `bw_bursts` | burst model with $p_b = 0.002$, scale = $12$ |
| $\eta_{\text{bw}}$ | `bw_noise` | $\mathcal{N}(0,\ 9)$ |

### Final value (clipped to [5.0, 99.9])

$$\text{BW}(t) = \text{clip}\!\left(\text{BW}(t),\ 5.0,\ 99.9\right)$$

### Code Reference (`enrich_6g.py`, lines 121–126)

```python
bw_base_signal = profile["bw_base"] + user_norm * 20
bw_util_pct = bw_base_signal + bw_bursts + bw_noise
bw_util_pct = clip(bw_util_pct, 5.0, 99.9)
```

---

## 9. CPU Utilization — Correlated with Bandwidth

**Origin:** Empirically motivated: packet processing load scales with network I/O. Linear coupling via normalized BW signal.

### Formula

$$\text{CPU}(t) = B_{\text{cpu}} + 18 \cdot \widehat{\text{BW}}(t) + S_{\text{burst,cpu}}(t) + \eta_{\text{cpu}}(t)$$

Where:

| Symbol | Code variable | Value |
|---|---|---|
| $B_{\text{cpu}}$ | `cpu_base` | slice-type dependent |
| $\widehat{\text{BW}}(t)$ | `bw_norm` | min-max normalized bandwidth |
| $18$ | — | coupling weight (% CPU range driven by BW) |
| $S_{\text{burst,cpu}}$ | `cpu_bursts` | burst model with $p_b = 0.0025$, scale = $10$ |
| $\eta_{\text{cpu}}$ | `cpu_noise` | $\mathcal{N}(0,\ 6.25)$ |

### Final value (clipped to [5.0, 99.9])

$$\text{CPU}(t) = \text{clip}\!\left(\text{CPU}(t),\ 5.0,\ 99.9\right)$$

### Code Reference (`enrich_6g.py`, lines 129–134)

```python
cpu_base_signal = profile["cpu_base"] + bw_norm * 18
cpu_util_pct = cpu_base_signal + cpu_bursts + cpu_noise
cpu_util_pct = clip(cpu_util_pct, 5.0, 99.9)
```

---

## 10. Memory Utilization — Semi-Independent Slow Drift

**Origin:** Memory usage in network functions typically has a slower, smoother profile than CPU (due to buffering/caching). Modeled with a low-amplitude sinusoid and low-variance noise, offset in phase.

### Formula

$$\text{MEM}(t) = B_{\text{mem}} + 8 \cdot \sin\!\left(\omega t - \frac{\pi}{2} + \varphi + 0.8\right) + \eta_{\text{mem}}(t)$$

Where:

| Symbol | Code variable | Value |
|---|---|---|
| $B_{\text{mem}}$ | `mem_base` | slice-type dependent |
| $8$ | — | fixed amplitude (smaller than other signals) |
| $\varphi + 0.8$ | `phase + 0.8` | additional 0.8 rad offset for decorrelation from users |
| $\eta_{\text{mem}}$ | `mem_noise` | $\mathcal{N}(0,\ 2.25)$ |

### Final value (clipped to [10.0, 99.9])

$$\text{MEM}(t) = \text{clip}\!\left(\text{MEM}(t),\ 10.0,\ 99.9\right)$$

### Code Reference (`enrich_6g.py`, lines 137–140)

```python
mem_curve = daily_curve(n, amplitude=8.0, phase_shift=phase + 0.8)
mem_util_pct = profile["mem_base"] + mem_curve + mem_noise
mem_util_pct = clip(mem_util_pct, 10.0, 99.9)
```

---

## 11. Queue Length — Derived from CPU Excess

**Origin:** Inspired by **Little's Law** and queueing theory (M/M/1 queue): when server utilization exceeds a saturation threshold, queue depth grows approximately linearly with excess load.

### Formula

$$Q(t) = 0.8 \cdot \max\!\left(\text{CPU}(t) - 75,\ 0\right) + S_{\text{burst},q}(t) + \eta_q(t)$$

Where:

| Symbol | Code variable | Meaning |
|---|---|---|
| $75$ | — | CPU saturation threshold (%); queue starts growing above this |
| $0.8$ | — | linear coupling coefficient |
| $\max(\cdot, 0)$ | `np.maximum(cpu_util_pct - 75, 0)` | ReLU-like activation — zero below threshold |
| $S_{\text{burst},q}$ | random_bursts | $p_b = 0.003$, scale = $8$ |
| $\eta_q$ | gaussian_noise | $\mathcal{N}(0,\ 1)$ |

### Final value (clipped, integer)

$$Q(t) = \left\lfloor \text{clip}\!\left(Q(t),\ 0,\ 50\right) \right\rfloor$$

### Little's Law Context

Little's Law states: $L = \lambda W$ (mean queue length = arrival rate × mean wait time). The linear proportionality between CPU excess and queue length is consistent with this relationship when arrival rate is assumed constant and service rate degrades proportionally past the threshold.

### Code Reference (`enrich_6g.py`, lines 143–146)

```python
cpu_excess = np.maximum(cpu_util_pct - 75, 0)
queue_len = (cpu_excess * 0.8 + random_bursts(...) + gaussian_noise(...))
queue_len = clip(queue_len, 0, 50).astype(int)
```

---

## 12. Congestion Flag — Multi-Condition Binary Classifier

**Origin:** Threshold-based anomaly detection / binary labeling — standard in network operations and SLA (Service Level Agreement) monitoring.

### Formula (Boolean OR of three thresholds)

$$\text{congestion\_flag}(t) = \mathbf{1}\!\left[\text{CPU}(t) > 85\right] \lor \mathbf{1}\!\left[\text{BW}(t) > 90\right] \lor \mathbf{1}\!\left[Q(t) > 20\right]$$

Formally:

$$\text{congestion\_flag}(t) = \begin{cases} 1 & \text{if } \text{CPU}(t) > 85 \\ 1 & \text{if } \text{BW}(t) > 90 \\ 1 & \text{if } Q(t) > 20 \\ 0 & \text{otherwise} \end{cases}$$

### Threshold Origin

| Threshold | Value | Rationale |
|---|---|---|
| CPU > 85% | 85 | Industry-standard CPU overload boundary (consistent with ETSI NFV-INF 010) |
| BW > 90% | 90 | Near-saturation for RF/transport links (3GPP TS 28.552 KPI guidance) |
| Queue > 20 | 20 | Empirical: beyond ~20 packets queued, latency degrades non-linearly |

### Code Reference (`enrich_6g.py`, lines 149–153)

```python
congestion_flag = (
    (cpu_util_pct > 85) |
    (bw_util_pct > 90)  |
    (queue_len > 20)
).astype(int)
```

---

## 13. Clipping (Signal Bounding)

**Origin:** Hard saturation / value clamping — standard signal processing operation to enforce physical and operational constraints.

### Formula

$$\text{clip}(x, lo, hi) = \max\!\left(lo,\ \min\!\left(hi,\ x\right)\right)$$

Equivalently (element-wise for arrays):

$$\text{clip}(x, lo, hi) = \begin{cases} lo & x < lo \\ x & lo \leq x \leq hi \\ hi & x > hi \end{cases}$$

### Applied Bounds per Signal

| Signal | Lower bound ($lo$) | Upper bound ($hi$) |
|---|---|---|
| `active_users` | $1$ | $U_{\text{base}} + 1.2 \cdot \text{user\_scale}$ |
| `bw_util_pct` | $5.0$ | $99.9$ |
| `cpu_util_pct` | $5.0$ | $99.9$ |
| `mem_util_pct` | $10.0$ | $99.9$ |
| `queue_len` | $0$ | $50$ |

### Code Reference (`enrich_6g.py`, line 95)

```python
return np.clip(arr, lo, hi)
```

---

## 14. Timestamp Generation

**Origin:** Uniform discrete-time sampling — standard for time-series databases and network monitoring systems (1-minute SNMP polling interval is an industry standard, per RFC 1157).

### Formula

$$\text{timestamp}_k = t_0 + i \cdot n_{\text{min}} \cdot \Delta t + k \cdot \Delta t, \quad k = 0, 1, \ldots, n-1$$

Where:

| Symbol | Value | Meaning |
|---|---|---|
| $t_0$ | `2025-01-01 00:00:00` | Global start time |
| $i$ | `instance_idx` | Instance index (0-based) |
| $n_{\text{min}}$ | $1440$ | Minutes per instance (1 day) |
| $\Delta t$ | 1 minute | Sampling interval |

Each instance's time range is non-overlapping: instance $i$ covers the window $[t_0 + i \cdot 1440\ \text{min},\ t_0 + (i+1) \cdot 1440\ \text{min} - 1]$.

### Code Reference (`enrich_6g.py`, lines 156–161)

```python
global_offset_min = instance_idx * n
timestamps = pd.date_range(
    start=START_TIME + pd.Timedelta(minutes=global_offset_min),
    periods=n,
    freq="1min",
)
```

---

## 15. Summary Metrics (Dashboard — `app.py`)

These are computed from the filtered slice time window and displayed as KPI cards.

### Mean (Arithmetic Average)

$$\bar{X} = \frac{1}{N} \sum_{t=1}^{N} X(t)$$

Applied to: `cpu_util_pct`, `bw_util_pct`, `mem_util_pct`, `active_users`.

### Congestion Rate

$$\text{Congestion Rate} = \frac{1}{N} \sum_{t=1}^{N} \text{congestion\_flag}(t) \times 100\%$$

This is the arithmetic mean of the binary congestion flag, equivalent to the **proportion of congested minutes** in the selected window.

### Maximum

$$Q_{\max} = \max_{t \in [1, N]} Q(t)$$

Applied to: `queue_len`.

### Code Reference (`app.py`, lines 163–168)

```python
avg_cpu = df["cpu_util_pct"].mean()
avg_bw  = df["bw_util_pct"].mean()
pct_congested = df["congestion_flag"].mean() * 100
max_queue = int(df["queue_len"].max())
avg_mem  = df["mem_util_pct"].mean()
avg_users = df["active_users"].mean()
```

---

## 16. Signal Correlation Chain Diagram

```
                    ┌────────────────────────────────────────────────────────────┐
                    │              SIGNAL GENERATION PIPELINE                    │
                    └────────────────────────────────────────────────────────────┘

  Sinusoid(ω,A,φ)  ──┐
  RandomBursts(p,b) ──┼──► active_users(t) ──min-max normalize──► user_norm(t)
  GaussianNoise(σ)  ──┘                                                │
                                                                       │ × 20
  bw_base                                                              ▼
  GaussianNoise(σ)  ──┐                                    bw_base_signal(t)
  RandomBursts(p,b) ──┼──────────────────────────────────► bw_util_pct(t)
                      └──────────────────────────────────► (clip [5, 99.9])
                                                                       │
                                                           min-max normalize
                                                                       │ × 18
  cpu_base                                                             ▼
  GaussianNoise(σ)  ──┐                                    cpu_base_signal(t)
  RandomBursts(p,b) ──┼──────────────────────────────────► cpu_util_pct(t)
                      └──────────────────────────────────► (clip [5, 99.9])
                                                                       │
                                              max(CPU - 75, 0)  ×  0.8
                                                                       ▼
  RandomBursts(p,b) ──┐                                       queue_len(t)
  GaussianNoise(σ)  ──┘                                       (clip [0, 50])

  Sinusoid(ω,8,φ+0.8) ──┐
  GaussianNoise(σ=1.5) ──┼──► mem_util_pct(t)   [semi-independent]
  mem_base               ┘    (clip [10, 99.9])

  ┌───────────────────────────────────────────────────────────────┐
  │  congestion_flag(t) = (CPU > 85) OR (BW > 90) OR (Q > 20)   │
  └───────────────────────────────────────────────────────────────┘
```

---

## Appendix A — Slice-Type Baseline Parameters

| Slice Type | `cpu_base` | `mem_base` | `bw_base` | `user_base` | `user_scale` |
|---|---|---|---|---|---|
| ERLLC  | 55 | 50 | 60 | 80  | 40  |
| umMTC  | 35 | 40 | 30 | 200 | 150 |
| MBRLLC | 70 | 65 | 75 | 60  | 35  |
| mURLLC | 60 | 55 | 65 | 50  | 30  |
| feMBB  | 80 | 70 | 85 | 300 | 200 |

**Slice type meanings:**
- **ERLLC** — Enhanced Reliable Low-Latency Communication
- **umMTC** — Ultra-massive Machine-Type Communication
- **MBRLLC** — Mobile Broadband Reliable Low-Latency Communication
- **mURLLC** — Massive Ultra-Reliable Low-Latency Communication
- **feMBB** — Further enhanced Mobile Broadband

---

## Appendix B — Global Simulation Constants

| Constant | Value | Meaning |
|---|---|---|
| `RANDOM_SEED` | 42 | Numpy RNG seed (reproducibility) |
| `INSTANCES_PER_SLICE_TYPE` | 3 | Number of slice instances per type |
| `MINUTES_PER_INSTANCE` | 1440 | Duration per instance = 24 h |
| `START_TIME` | 2025-01-01 00:00:00 | Time-series origin |
| Total rows generated | $5 \times 3 \times 1440 = 21{,}600$ | 5 types × 3 instances × 1440 min |

---

*Generated from source analysis of `enrich_6g.py` and `app.py` — Simulation-Forecasting project.*
