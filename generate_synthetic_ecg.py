"""
generate_synthetic_ecg.py
─────────────────────────
Synthesise realistic 12-lead ECG signals for six rhythm/morphology classes
and save them as a compressed NumPy archive (ecg_data.npz).

Output shape:
    signals : (N, 12, 5000)   float32
    labels  : (N,)             int64    [0..5]
    classes : list of str      CLASS_NAMES from config

Generation approach
───────────────────
Every ECG is built beat-by-beat at 500 Hz.  Each beat contains:
  • P wave  – Gaussian envelope
  • QRS     – narrow Gaussian (positive) with negative Q/S flanks
  • T wave  – broad Gaussian

The amplitude of each component is scaled per-lead using a simplified
lead-projection model (axis projection) so that the twelve waveforms look
clinically plausible rather than identical copies.

Class-specific modifications are layered on top of the Normal template:
  AFIB     – irregular RR, suppressed P waves, fibrillatory baseline
  STEMI    – ST elevation in inferior or anterior lead group
  LBBB     – wide notched QRS, R in V5-V6, rS in V1-V3
  VT       – rapid rate >150 bpm, wide QRS, monophasic pattern
  AV_Block – prolonged PR or randomly dropped beats (3rd-degree pattern)
"""

import os
import time
import numpy as np

from config import (
    SAMPLE_RATE, DURATION_SEC, SIGNAL_LENGTH, NUM_LEADS, LEAD_NAMES,
    NUM_CLASSES, CLASS_NAMES, NUM_SAMPLES, DATA_FILE, RANDOM_SEED,
)

rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_pulse(t, center, sigma, amplitude):
    """Scalar or array Gaussian bump."""
    return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _lead_scale(lead_idx, axis_deg=60.0):
    """
    Rough projection of the cardiac electrical axis onto each lead direction.
    Positive = upright deflection in that lead.
    """
    # Lead directions (degrees from lead I):
    directions = np.array([
        0,    # I
        60,   # II
        120,  # III
        -150, # aVR
        -30,  # aVL
        90,   # aVF
        -135, # V1   (negative for septal activation)
        -100, # V2
        -60,  # V3
        -20,  # V4
        20,   # V5
        50,   # V6
    ], dtype=float)
    proj = np.cos(np.radians(directions[lead_idx] - axis_deg))
    return proj


# ─────────────────────────────────────────────────────────────────────────────
#  Normal ECG generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_normal_beat(
    t_start,          # seconds: start of beat window
    rr_interval,      # seconds: R-to-R interval
    pr_interval,      # seconds: PR interval (P onset to QRS onset)
    qrs_duration,     # seconds
    qt_interval,      # seconds
    lead_idx,
    axis_deg=60.0,
    noise_std=0.02,
):
    """Return (times, signal) arrays for one Normal beat."""
    fs = SAMPLE_RATE
    n_beat = int(rr_interval * fs)
    t = np.linspace(t_start, t_start + rr_interval, n_beat, endpoint=False)

    ls = _lead_scale(lead_idx, axis_deg)

    # P wave — centred 60 % of the way through the PR segment
    p_center = t_start + pr_interval * 0.6
    p_sigma  = 0.025
    p_amp    = 0.15 * abs(ls) + 0.02

    # QRS — R peak at t_start + pr_interval
    qrs_center = t_start + pr_interval
    r_sigma    = qrs_duration / 4.0
    r_amp      = 1.0 * ls          # main R wave follows axis projection

    # Q wave (negative, before R)
    q_center = qrs_center - qrs_duration * 0.4
    q_sigma  = qrs_duration / 6.0
    q_amp    = -0.1 * abs(ls)

    # S wave (negative, after R)
    s_center = qrs_center + qrs_duration * 0.5
    s_sigma  = qrs_duration / 6.0
    s_amp    = -0.15 * abs(ls)

    # T wave — centred at qrs + 60 % of ST+T interval
    st_t_dur  = qt_interval - qrs_duration
    t_center  = qrs_center + qrs_duration + st_t_dur * 0.55
    t_sigma   = st_t_dur * 0.25
    t_amp     = 0.35 * ls

    sig = (
        gaussian_pulse(t, p_center, p_sigma, p_amp)
        + gaussian_pulse(t, qrs_center, r_sigma, r_amp)
        + gaussian_pulse(t, q_center,   q_sigma, q_amp)
        + gaussian_pulse(t, s_center,   s_sigma, s_amp)
        + gaussian_pulse(t, t_center,   t_sigma, t_amp)
    )
    return t, sig


# ─────────────────────────────────────────────────────────────────────────────
#  Per-class ECG builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_normal(lead_idx):
    """60–100 bpm, all intervals within normal range."""
    fs = SAMPLE_RATE
    hr  = rng.uniform(60, 100)
    rr  = 60.0 / hr
    pr  = rng.uniform(0.16, 0.20)
    qrs = rng.uniform(0.08, 0.10)
    qt  = rng.uniform(0.36, 0.44)
    axis = rng.uniform(0, 90)

    signal = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
    t_cur  = 0.0
    while t_cur < DURATION_SEC - rr:
        _, seg = _make_normal_beat(t_cur, rr, pr, qrs, qt, lead_idx, axis)
        i_start = int(t_cur * fs)
        i_end   = i_start + len(seg)
        if i_end > SIGNAL_LENGTH:
            break
        signal[i_start:i_end] += seg.astype(np.float32)
        t_cur += rr

    noise = rng.normal(0, 0.015, SIGNAL_LENGTH).astype(np.float32)
    return signal + noise


def _build_afib(lead_idx):
    """
    Irregular RR (350–900 ms), absent P waves, fibrillatory baseline.
    """
    fs  = SAMPLE_RATE
    qrs = rng.uniform(0.08, 0.10)
    qt  = rng.uniform(0.32, 0.40)
    axis = rng.uniform(0, 90)
    pr  = 0.0  # no true P wave; QRS onset used directly

    signal = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
    t_cur  = rng.uniform(0.1, 0.4)
    while t_cur < DURATION_SEC - 0.9:
        rr = rng.uniform(0.35, 0.90)
        ls = _lead_scale(lead_idx, axis)

        # QRS only (no P wave)
        n_beat = int(rr * fs)
        t = np.linspace(t_cur, t_cur + rr, n_beat, endpoint=False)
        qrs_center = t_cur + rng.uniform(0.05, 0.12)
        r_sigma    = qrs / 4.0
        r_amp      = 1.0 * ls
        q_amp      = -0.08 * abs(ls)
        s_amp      = -0.12 * abs(ls)
        q_center   = qrs_center - qrs * 0.4
        s_center   = qrs_center + qrs * 0.5

        # T wave
        st_t = qt - qrs
        t_center = qrs_center + qrs + st_t * 0.55
        t_sigma  = st_t * 0.25
        t_amp    = 0.3 * ls

        seg = (
            gaussian_pulse(t, qrs_center, r_sigma, r_amp)
            + gaussian_pulse(t, q_center, qrs / 6.0, q_amp)
            + gaussian_pulse(t, s_center, qrs / 6.0, s_amp)
            + gaussian_pulse(t, t_center, t_sigma, t_amp)
        )
        i_start = int(t_cur * fs)
        i_end   = i_start + len(seg)
        if i_end > SIGNAL_LENGTH:
            break
        signal[i_start:i_end] += seg.astype(np.float32)
        t_cur += rr

    # Fibrillatory baseline (fine irregular oscillations ~350–600 Hz look-alike
    # simulated with sum of slightly displaced sinusoids)
    t_axis = np.linspace(0, DURATION_SEC, SIGNAL_LENGTH)
    fib = sum(
        0.03 * np.sin(2 * np.pi * f * t_axis + rng.uniform(0, 2 * np.pi))
        for f in rng.uniform(4, 9, 6)
    )
    noise = rng.normal(0, 0.015, SIGNAL_LENGTH)
    return (signal + fib + noise).astype(np.float32)


def _build_stemi(lead_idx):
    """
    ST elevation in inferior (II, III, aVF → indices 1,2,5) or
    anterior (V1-V4 → indices 6-9) leads; reciprocal depression elsewhere.
    """
    fs   = SAMPLE_RATE
    hr   = rng.uniform(60, 110)
    rr   = 60.0 / hr
    pr   = rng.uniform(0.16, 0.20)
    qrs  = rng.uniform(0.08, 0.10)
    qt   = rng.uniform(0.36, 0.44)
    axis = rng.uniform(60, 90)

    # Randomly choose inferior or anterior STEMI
    inferior_leads  = {1, 2, 5}
    anterior_leads  = {6, 7, 8, 9}
    stemi_type = rng.choice(["inferior", "anterior"])
    elevation_leads  = inferior_leads if stemi_type == "inferior" else anterior_leads
    reciprocal_leads = anterior_leads if stemi_type == "inferior" else inferior_leads

    st_elevation  =  rng.uniform(0.15, 0.40)
    st_depression = -rng.uniform(0.05, 0.15)

    if lead_idx in elevation_leads:
        st_offset = st_elevation
    elif lead_idx in reciprocal_leads:
        st_offset = st_depression
    else:
        st_offset = 0.0

    signal = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
    t_cur  = 0.0
    while t_cur < DURATION_SEC - rr:
        n_beat     = int(rr * fs)
        t          = np.linspace(t_cur, t_cur + rr, n_beat, endpoint=False)
        ls         = _lead_scale(lead_idx, axis)
        qrs_center = t_cur + pr
        r_sigma    = qrs / 4.0
        q_center   = qrs_center - qrs * 0.4
        s_center   = qrs_center + qrs * 0.5
        st_dur     = qt - qrs
        t_center   = qrs_center + qrs + st_dur * 0.55
        t_sigma    = st_dur * 0.25

        # Tombstone T in elevation leads (merged ST-T)
        if lead_idx in elevation_leads:
            t_amp = 0.5 * abs(ls)
        else:
            t_amp = 0.35 * ls

        seg = (
            gaussian_pulse(t, qrs_center, r_sigma, 1.0 * ls)
            + gaussian_pulse(t, q_center, qrs / 6.0, -0.1 * abs(ls))
            + gaussian_pulse(t, s_center, qrs / 6.0, -0.15 * abs(ls))
            + gaussian_pulse(t, t_center, t_sigma,   t_amp)
            + st_offset  # constant ST shift on every sample of this beat
        )
        # Blend ST offset smoothly: apply only in the ST segment window
        st_start = int((qrs_center + qrs) * fs) - int(t_cur * fs)
        st_end   = int((qrs_center + qt)  * fs) - int(t_cur * fs)
        seg_mod  = seg.copy()
        if 0 < st_start < len(seg_mod):
            seg_mod[:st_start] -= st_offset  # remove constant from QRS
        if 0 < st_end < len(seg_mod):
            seg_mod[st_end:]   -= st_offset  # remove from after T

        i_start = int(t_cur * fs)
        i_end   = i_start + len(seg_mod)
        if i_end > SIGNAL_LENGTH:
            break
        signal[i_start:i_end] += seg_mod.astype(np.float32)
        t_cur += rr

    noise = rng.normal(0, 0.015, SIGNAL_LENGTH)
    return (signal + noise).astype(np.float32)


def _build_lbbb(lead_idx):
    """
    Left Bundle Branch Block:
    • QRS > 0.12 s (wide, 0.14-0.18 s)
    • Notched R ('M'-shape) in V5, V6 (indices 10, 11) and I (0), aVL (4)
    • rS pattern in V1-V3 (indices 6, 7, 8): small r, deep S
    • No septal Q in lateral leads
    • Secondary ST-T discordance
    """
    fs   = SAMPLE_RATE
    hr   = rng.uniform(55, 100)
    rr   = 60.0 / hr
    pr   = rng.uniform(0.16, 0.22)
    qrs  = rng.uniform(0.14, 0.18)   # wide QRS
    qt   = rng.uniform(0.40, 0.50)
    axis = rng.uniform(-30, 30)      # often left-axis deviation

    lateral_leads = {0, 4, 10, 11}   # I, aVL, V5, V6  — notched R
    rs_leads      = {6, 7, 8}        # V1-V3 — rS

    signal = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
    t_cur  = 0.0
    while t_cur < DURATION_SEC - rr:
        n_beat     = int(rr * fs)
        t          = np.linspace(t_cur, t_cur + rr, n_beat, endpoint=False)
        ls         = _lead_scale(lead_idx, axis)
        qrs_center = t_cur + pr
        st_dur     = qt - qrs
        t_center   = qrs_center + qrs + st_dur * 0.55
        t_sigma    = st_dur * 0.28

        if lead_idx in lateral_leads:
            # Notched (bifid) R: two Gaussian peaks separated by ~half QRS
            r1_center = qrs_center - qrs * 0.15
            r2_center = qrs_center + qrs * 0.15
            r_sigma   = qrs / 6.0
            seg = (
                gaussian_pulse(t, r1_center, r_sigma,  0.8 * abs(ls))
                + gaussian_pulse(t, r2_center, r_sigma, 0.8 * abs(ls))
                # Discordant T (inverted relative to QRS)
                + gaussian_pulse(t, t_center,  t_sigma, -0.25 * abs(ls))
            )
        elif lead_idx in rs_leads:
            # rS: small positive r, large negative S
            r_center = qrs_center - qrs * 0.2
            s_center = qrs_center + qrs * 0.2
            seg = (
                gaussian_pulse(t, r_center, qrs / 8.0,  0.15)
                + gaussian_pulse(t, s_center, qrs / 4.0, -0.9)
                + gaussian_pulse(t, t_center, t_sigma,    0.20)
            )
        else:
            # Other leads: broad QS or RS
            r_sigma = qrs / 3.5
            seg = (
                gaussian_pulse(t, qrs_center, r_sigma, 0.6 * ls)
                + gaussian_pulse(t, t_center,  t_sigma, -0.2 * ls)
            )

        i_start = int(t_cur * fs)
        i_end   = i_start + len(seg)
        if i_end > SIGNAL_LENGTH:
            break
        signal[i_start:i_end] += seg.astype(np.float32)
        t_cur += rr

    noise = rng.normal(0, 0.015, SIGNAL_LENGTH)
    return (signal + noise).astype(np.float32)


def _build_vt(lead_idx):
    """
    Ventricular Tachycardia:
    • Rate 150–220 bpm
    • Wide QRS (0.14–0.20 s)
    • Monophasic (positive or negative) across most leads
    • Regular RR
    • No identifiable P waves
    """
    fs   = SAMPLE_RATE
    hr   = rng.uniform(150, 220)
    rr   = 60.0 / hr
    qrs  = rng.uniform(0.14, 0.20)
    qt   = rng.uniform(0.28, 0.38)
    axis = rng.uniform(90, 180)      # right-axis or extreme-axis

    signal = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
    t_cur  = rng.uniform(0.0, 0.1)
    while t_cur < DURATION_SEC - rr:
        n_beat     = int(rr * fs)
        t          = np.linspace(t_cur, t_cur + rr, n_beat, endpoint=False)
        ls         = _lead_scale(lead_idx, axis)
        qrs_center = t_cur + rr * 0.25
        r_sigma    = qrs / 3.0   # broad QRS
        st_dur     = max(qt - qrs, 0.05)
        t_center   = qrs_center + qrs + st_dur * 0.5
        t_sigma    = st_dur * 0.3
        # Predominantly monophasic
        r_amp = 1.2 * ls
        t_amp = -0.4 * ls    # discordant T

        seg = (
            gaussian_pulse(t, qrs_center, r_sigma, r_amp)
            + gaussian_pulse(t, t_center,  t_sigma, t_amp)
        )
        i_start = int(t_cur * fs)
        i_end   = i_start + len(seg)
        if i_end > SIGNAL_LENGTH:
            break
        signal[i_start:i_end] += seg.astype(np.float32)
        t_cur += rr

    noise = rng.normal(0, 0.012, SIGNAL_LENGTH)
    return (signal + noise).astype(np.float32)


def _build_av_block(lead_idx):
    """
    Complete (3rd-degree) AV Block:
    • Atrial rate 60–100 bpm (independent P waves)
    • Ventricular rate 30–45 bpm (escape rhythm, wide QRS)
    • No fixed PR relationship
    """
    fs       = SAMPLE_RATE
    atrial_hr  = rng.uniform(60, 100)
    vent_hr    = rng.uniform(30, 45)
    pp         = 60.0 / atrial_hr   # P-P interval
    rr         = 60.0 / vent_hr     # R-R interval (escape)
    qrs        = rng.uniform(0.12, 0.16)  # escape QRS is moderately wide
    qt         = rng.uniform(0.40, 0.55)
    axis_p     = rng.uniform(30, 70)
    axis_qrs   = rng.uniform(-30, 30)

    signal = np.zeros(SIGNAL_LENGTH, dtype=np.float32)

    # Independent P waves
    t_p = rng.uniform(0.0, pp)
    while t_p < DURATION_SEC:
        ls_p = _lead_scale(lead_idx, axis_p)
        p_sigma = 0.025
        n_p = int(pp * fs)
        t   = np.linspace(t_p, t_p + pp, n_p, endpoint=False)
        seg = gaussian_pulse(t, t_p + pp * 0.1, p_sigma, 0.15 * abs(ls_p))
        i_start = int(t_p * fs)
        i_end   = i_start + len(seg)
        if i_end <= SIGNAL_LENGTH:
            signal[i_start:i_end] += seg.astype(np.float32)
        t_p += pp

    # Independent QRS (escape beats)
    t_r = rng.uniform(0.0, rr)
    while t_r < DURATION_SEC - rr:
        ls = _lead_scale(lead_idx, axis_qrs)
        n_beat     = int(rr * fs)
        t          = np.linspace(t_r, t_r + rr, n_beat, endpoint=False)
        qrs_center = t_r + rr * 0.2
        r_sigma    = qrs / 3.5
        st_dur     = qt - qrs
        t_center   = qrs_center + qrs + st_dur * 0.55
        t_sigma    = st_dur * 0.28

        seg = (
            gaussian_pulse(t, qrs_center, r_sigma, 0.9 * ls)
            + gaussian_pulse(t, qrs_center - qrs * 0.4, qrs / 6.0, -0.08 * abs(ls))
            + gaussian_pulse(t, qrs_center + qrs * 0.5, qrs / 6.0, -0.1  * abs(ls))
            + gaussian_pulse(t, t_center,  t_sigma, 0.3 * ls)
        )
        i_start = int(t_r * fs)
        i_end   = i_start + len(seg)
        if i_end > SIGNAL_LENGTH:
            break
        signal[i_start:i_end] += seg.astype(np.float32)
        t_r += rr

    noise = rng.normal(0, 0.015, SIGNAL_LENGTH)
    return (signal + noise).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Builder dispatch
# ─────────────────────────────────────────────────────────────────────────────

_BUILDERS = {
    0: _build_normal,
    1: _build_afib,
    2: _build_stemi,
    3: _build_lbbb,
    4: _build_vt,
    5: _build_av_block,
}


def generate_ecg(class_idx: int) -> np.ndarray:
    """Return one ECG of shape (12, 5000) for the given class index."""
    build_fn = _BUILDERS[class_idx]
    leads = np.stack([build_fn(lead_idx) for lead_idx in range(NUM_LEADS)], axis=0)
    return leads.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Generating {NUM_SAMPLES} synthetic ECGs …")
    print(f"  Signal: {NUM_LEADS} leads × {SIGNAL_LENGTH} samples @ {SAMPLE_RATE} Hz = {DURATION_SEC}s")
    print(f"  Classes: {CLASS_NAMES}")

    n_per_class = NUM_SAMPLES // NUM_CLASSES
    signals_list = []
    labels_list  = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        print(f"  [{cls_idx+1}/{NUM_CLASSES}] {cls_name:15s} — {n_per_class} samples …", end=" ", flush=True)
        t_cls = time.time()
        for _ in range(n_per_class):
            signals_list.append(generate_ecg(cls_idx))
            labels_list.append(cls_idx)
        print(f"{time.time() - t_cls:.1f}s")

    signals = np.stack(signals_list, axis=0)   # (N, 12, 5000)
    labels  = np.array(labels_list, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(labels))
    signals = signals[idx]
    labels  = labels[idx]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE)
    np.savez_compressed(
        out_path,
        signals=signals,
        labels=labels,
        class_names=np.array(CLASS_NAMES),
        lead_names=np.array(LEAD_NAMES),
        sample_rate=np.int32(SAMPLE_RATE),
        duration_sec=np.int32(DURATION_SEC),
    )

    elapsed = time.time() - t0
    print(f"\nSaved  : {out_path}")
    print(f"Shape  : signals={signals.shape}  labels={labels.shape}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Class distribution: { {CLASS_NAMES[i]: int((labels==i).sum()) for i in range(NUM_CLASSES)} }")


if __name__ == "__main__":
    main()
