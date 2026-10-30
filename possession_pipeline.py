"""Production-grade Controlled Possession Pipeline for More than a Hack 2027.

Combines:
  1. Ball Trajectory Imputation & Velocity Modeling:
     - Gaps <= 30 frames are interpolated with kinematic bounds (v <= 35 m/s).
     - Instantaneous ball velocity is computed to distinguish live passes/shots from placed dead balls.
  2. Physical & Collective Stoppage Detection (Dead Ball / 'D'):
     - Exact pitch boundary projection: ball outside field margins (x < -0.2, x > 105.5, y < -0.2, y > 68.2) is physical dead ball.
     - Collective player velocity collapse: rolling average player speed dropping below 1.20 m/s indicates game stoppage.
     - Dead ball candidate persistence filtering: requires sustained stoppage condition (>= 50 frames / 2.0s) to avoid spurious splits during live build-up.
  3. Controlled Possession State Machine with Inertia & Hysteresis:
     - Follows official challenge rules: loose balls and passes in flight maintain possession with the last team in control.
     - Turnover (H -> A or A -> H) requires sustained opposing control touches (streak >= 12 frames within control distance 2.5m).
     - Restarts from dead ball require in-bounds ball movement/kick or proximity control.
  4. Micro-Spell Glitch Filtering:
     - Rapid transient deflections (< 25 frames / 1s) are cleanly smoothed back into the true possession spell.

Usage:
    python possession_pipeline.py
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames

RESULTS_DIR = os.path.join(HERE, "evaluation", "results")


def build_controlled_possession(game=GAME, data_dir=None,
                                speed_thresh=1.20,
                                speed_min_len=50,
                                control_dist=2.5,
                                turnover_streak=12,
                                glitch_len=25):
    """Run the complete possession detection pipeline on the match tracking data."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch_len, pitch_wid = GAMES[game]["pitch"]

    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    track_teams = np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")], dtype=np.int8)

    print(f"Reading {os.path.basename(raw_path)} ({n_frames} frames)...")
    ball_pos = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.zeros(n_frames, dtype=np.float32)
    min_dist_h = np.full(n_frames, 999.0, dtype=np.float32)
    min_dist_a = np.full(n_frames, 999.0, dtype=np.float32)
    mean_speed = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid >= n_frames:
                continue

            ball = parts[2]
            has_ball = (ball[0] != "N")
            if has_ball:
                comma = ball.index(",")
                bx = float(ball[:comma]) * pitch_len
                by = float(ball[comma + 1:ball.index(",", comma + 1)]) * pitch_wid
                ball_pos[fid, 0] = bx
                ball_pos[fid, 1] = by

            speeds = []
            dh_min, da_min = 999.0, 999.0
            for slot, grp in enumerate(parts[1].split(";")):
                if grp[0] == "N":
                    continue
                comma = grp.index(",")
                px = float(grp[:comma]) * pitch_len
                comma2 = grp.index(",", comma + 1)
                py = float(grp[comma + 1:comma2]) * pitch_wid
                s_str = grp[comma2 + 1:]
                if s_str != "NaN":
                    speeds.append(float(s_str))

                if has_ball:
                    d = np.hypot(px - bx, py - by)
                    if track_teams[slot] == 0:
                        if d < dh_min:
                            dh_min = d
                    else:
                        if d < da_min:
                            da_min = d

            if speeds:
                mean_speed[fid] = np.mean(speeds)
            min_dist_h[fid] = dh_min
            min_dist_a[fid] = da_min

    print("Step 1: Kinematic ball trajectory interpolation...")
    valid = ~np.isnan(ball_pos[:, 0])
    valid_idx = np.flatnonzero(valid)
    for i in range(len(valid_idx) - 1):
        idx1, idx2 = valid_idx[i], valid_idx[i + 1]
        gap = idx2 - idx1 - 1
        if 0 < gap <= 30:
            pos1, pos2 = ball_pos[idx1], ball_pos[idx2]
            dt = (gap + 1) / 25.0
            if np.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1]) / dt <= 35.0:
                for k in range(1, gap + 1):
                    alpha = k / (gap + 1)
                    ball_pos[idx1 + k] = (1 - alpha) * pos1 + alpha * pos2

    # Compute ball speed (m/s)
    for f in range(n_frames - 1):
        if not np.isnan(ball_pos[f, 0]) and not np.isnan(ball_pos[f + 1, 0]):
            ball_speed[f] = np.hypot(ball_pos[f + 1, 0] - ball_pos[f, 0],
                                     ball_pos[f + 1, 1] - ball_pos[f, 1]) * 25.0

    print("Step 2: Collective velocity & physical boundary dead ball detection...")
    df_speed = pd.Series(mean_speed).interpolate().bfill().ffill()
    smooth_speed = df_speed.rolling(50, min_periods=1, center=True).mean().values

    # Physical pitch bounds check
    out_of_bounds = (ball_pos[:, 0] < -0.2) | (ball_pos[:, 0] > pitch_len + 0.2) | \
                    (ball_pos[:, 1] < -0.2) | (ball_pos[:, 1] > pitch_wid + 0.2)

    dead_candidate = out_of_bounds | ((smooth_speed < speed_thresh) & (np.minimum(min_dist_h, min_dist_a) > 2.0))

    dead_mask = np.zeros(n_frames, dtype=bool)
    in_dead = False
    start_d = 0
    for f in range(n_frames):
        if dead_candidate[f]:
            if not in_dead:
                in_dead = True
                start_d = f
        else:
            if in_dead:
                in_dead = False
                if f - start_d >= speed_min_len:
                    dead_mask[start_d:f] = True
    if in_dead and (n_frames - start_d >= speed_min_len):
        dead_mask[start_d:n_frames] = True

    print("Step 3: Controlled possession state machine with turnover hysteresis...")
    HOME, AWAY, DEAD = 0, 1, 2
    state = DEAD
    labels_out = np.full(n_frames, "D", dtype="<U1")

    h_control_count = 0
    a_control_count = 0

    for f in range(n_frames):
        if dead_mask[f]:
            state = DEAD
            h_control_count = 0
            a_control_count = 0
        else:
            dh = min_dist_h[f]
            da = min_dist_a[f]
            bs = ball_speed[f]
            in_field = not out_of_bounds[f] and not np.isnan(ball_pos[f, 0])

            h_touch = (dh < control_dist) and (dh < da)
            a_touch = (da < control_dist) and (da < dh)

            if h_touch:
                h_control_count += 1
                a_control_count = 0
            elif a_touch:
                a_control_count += 1
                h_control_count = 0
            else:
                h_control_count = max(0, h_control_count - 1)
                a_control_count = max(0, a_control_count - 1)

            if state == DEAD:
                if in_field and (bs > 2.5 or dh < 1.5 or da < 1.5):
                    if h_control_count >= 5 or (dh < 1.5 and dh < da):
                        state = HOME
                    elif a_control_count >= 5 or (da < 1.5 and da < dh):
                        state = AWAY
            elif state == HOME:
                if a_control_count >= turnover_streak and da < control_dist:
                    state = AWAY
            elif state == AWAY:
                if h_control_count >= turnover_streak and dh < control_dist:
                    state = HOME

        if state == HOME:
            labels_out[f] = "H"
        elif state == AWAY:
            labels_out[f] = "A"
        else:
            labels_out[f] = "D"

    print("Step 4: Glitch filtering...")
    runs = []
    cur_lbl = labels_out[0]
    start_f = 0
    for f in range(1, n_frames):
        if labels_out[f] != cur_lbl:
            runs.append((cur_lbl, start_f, f))
            cur_lbl = labels_out[f]
            start_f = f
    runs.append((cur_lbl, start_f, n_frames))

    cleaned = labels_out.copy()
    for i in range(1, len(runs) - 1):
        lbl, s, e = runs[i]
        length = e - s
        prev_lbl, ps, pe = runs[i - 1]
        next_lbl, ns, ne = runs[i + 1]
        if length < glitch_len and prev_lbl == next_lbl:
            cleaned[s:e] = prev_lbl

    return cleaned


def main():
    out_dir = os.path.join(HERE, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission_controlled_possession.csv")

    labels = build_controlled_possession()
    print(f"Writing {out_path} ...")
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(labels))

    print(f"Done! Saved {len(labels)} rows to {out_path}")
    print("Validating submission format...")
    os.system(f'"{sys.executable}" evaluation/validate_submission.py "{out_path}"')
    print("\nRunning evaluation on public ground truth...")
    os.system(f'"{sys.executable}" evaluation/run_evaluation.py')


if __name__ == "__main__":
    main()
