#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os 
import vr2p

# =========================
# Configuration
# =========================
DATA_PATH = "../DataSet.zarr"
OUT_PREFIX = "vr2p_extracted"

# =========================
# Load data
# =========================
data = vr2p.ExperimentData(DATA_PATH)

# =========================
# Marker definitions
# =========================
markers = [
    {"name": "Indicator", "position": [60, 100]},
    {"name": "R1", "position": [130, 150]},
    {"name": "R2", "position": [180, 200]},
    {"name": "Teleportation", "position": [1_000_000 - 1, 1_000_000 + 1]},
]

# =========================
# Containers
# =========================
F_big = []
spks_big = []
day_ind = []
reward_id = []
status = []
cue_set_id = []
selected_pos_big = []

# =========================
# Main loop
# =========================
for session_id in range(len(data.vr)):
    print(f"Analyzing {session_id}")
    F_session = data.signals.multi_session.Fns[session_id]
    if np.isnan(F_session[:]).any():
        continue

    vr = data.vr[session_id]
    trial = vr.trial.copy()
    lick = vr.lick

    position = vr.path.frame.copy().reset_index()

    # Merge trial metadata
    position = position.merge(
        trial[["set", "trial_number", "reward_id", "status"]],
        on="trial_number",
        how="left",
    )

    # Compute speed
    position["speed"] = position.vr2p.rolling_speed(
        window_size=100,
        ignore_threshold=7.5,
    )
    position.loc[position["interim_number"].notna(), "speed"] = 1000

    position_speed_filtered = position.loc[position["speed"] >= 0].copy()
    position_speed_filtered["has_lick"] = 0
    position_speed_filtered.loc[
        position_speed_filtered.frame.isin(lick.frame), "has_lick"
    ] = 1

    selected_trials = trial.loc[
        trial.set.isin(trial.set.unique()), "trial_number"
    ]

    # Teleportation marker
    position_speed_filtered.loc[
        position_speed_filtered.interim_number.isin(selected_trials[:-1]),
        "position",
    ] = 1_000_000

    selected_position = position_speed_filtered.loc[
        (position_speed_filtered.trial_number.isin(selected_trials[:-1]))
        | (position_speed_filtered.interim_number.isin(selected_trials[:-1])),
        [
            "set",
            "frame",
            "position",
            "trial_number",
            "interim_number",
            "period_number",
            "reward_id",
            "status",
            "speed",
            "has_lick",
        ],
    ].copy()

    # Label positions
    selected_position["position_marker"] = "Track"
    dist_name = ["Near", "Far"]

    for rid in (1, 2):
        for marker in markers:
            selected_position.loc[
                (selected_position.reward_id == rid)
                & selected_position.position.between(
                    marker["position"][0], marker["position"][1]
                ),
                "position_marker",
            ] = f"{marker['name']}-{dist_name[rid - 1]}"

    selected_position.loc[
        selected_position.interim_number.isin(selected_trials[:-1]),
        "position_marker",
    ] = "Teleportation"

    # Assign teleportation frames to next trial
    mask = selected_position["interim_number"].notna()
    selected_position.loc[mask, "trial_number"] = (
        selected_position.loc[mask, "interim_number"]
    )

    frames = selected_position["frame"].to_numpy()

    F_big.append(F_session[:, frames])
    spks_big.append(data.signals.multi_session.spks[session_id][:, frames])

    n_frames = len(frames)
    day_ind.append(np.full((n_frames, 1), session_id))

    reward_id.append(selected_position.reward_id.to_numpy())
    status.append(selected_position.status.to_numpy())
    cue_set_id.append(selected_position.set.to_numpy())
    selected_pos_big.append(selected_position)

# =========================
# Stack arrays
# =========================
F_big_array = np.hstack(F_big)
spks_big_array = np.hstack(spks_big)
day_ind_array = np.vstack(day_ind)
reward_id_array = np.hstack(reward_id)
status_array = np.hstack(status)
cue_set_id_array = np.hstack(cue_set_id)

# =========================
# Save numeric arrays
# =========================
np.savez_compressed(
    f"{OUT_PREFIX}_signals.npz",
    F_big_array=F_big_array,
    spks_big_array=spks_big_array,
    day_ind_array=day_ind_array,
    reward_id_array=reward_id_array,
    status_array=status_array,
    cue_set_id_array=cue_set_id_array,
)

# =========================
# Save selected_pos_big (compressed)
# =========================
parquet_dir = f"{OUT_PREFIX}_selected_pos_parquet"
os.makedirs(parquet_dir, exist_ok=True)

for i, df in enumerate(selected_pos_big):
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(
        table,
        f"{parquet_dir}/selected_pos_{i:03d}.parquet",
        compression="zstd",
    )

print("Extraction complete.")
print(f"Saved: {OUT_PREFIX}_signals.npz")
print(f"Saved: {parquet_dir}/selected_pos_*.parquet")
