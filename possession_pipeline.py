"""Production-grade Controlled Possession Pipeline for More than a Hack 2027.

Option A Enhanced Pipeline:
  1. Kinematic Ball Trajectory Imputation & Velocity Modeling:
     - Gaps <= 30 frames are interpolated under kinematic bounds (v <= 35 m/s).
     - Instantaneous ball velocity is computed to model passes, shots, and set-piece strikes.
  2. Pass-in-Flight & Teammate Reception Tracking:
     - Analyzes ball flight vectors when launched (v >= 3.5 m/s) from a player.
     - Detects completed passes between teammates (striker_team == receiver_team).
     - Locks transit frames to the passing team, preventing opponents near the flight path
       from triggering false turnovers (resolves the 'pass-in-flight' dilemma).
  3. Touchline & Stoppage Dead Ball ('D') Detection:
     - Physical boundary check: ball outside pitch margins (x < -0.2, x > 105.5, y < -0.2, y > 68.2)
       is marked as an out-of-bounds stoppage.
     - Open-field collective deceleration: when players slow down during fouls/stoppages
       (rolling mean speed < 1.20 m/s for >= 50 frames with ball unassigned).
     - Stationary ball protection: while in 'D', players walking near a stationary ball (v < 1.8 m/s)
       cannot exit dead ball until the ball is actively kicked (v >= 2.5 m/s) into play.
  4. Turnover Hysteresis & Inertia State Machine:
     - Maintains possession with the attacking team during contested, loose, or rolling balls.
     - Opponent turnover requires sustained, controlled proximity (streak >= 12 frames within 2.5m).
  5. Micro-Spell Glitch Filter:
     - Transient single-frame noise (< 25 frames / 1s) is smoothed into the surrounding state.

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
                                oob_min_len=25,
                                speed_thresh=1.20,
                                speed_min_len=50,
                                control_dist=2.5,
                                turnover_streak=12,
                                restart_min_speed=2.5,
                                glitch_len=25):
    """Run the Option A controlled possession pipeline on the match tracking data."""
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

    print("Step 1: Kinematic ball trajectory interpolation & speed calculation...")
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

    print("Step 2: Detecting pass-in-flight trajectories and teammate receptions...")
    pass_team = np.full(n_frames, -1, dtype=np.int8)
    in_flight = False
    flight_start = 0
    striker_team = -1

    out_of_bounds = (ball_pos[:, 0] < -0.2) | (ball_pos[:, 0] > pitch_len + 0.2) | \
                    (ball_pos[:, 1] < -0.2) | (ball_pos[:, 1] > pitch_wid + 0.2)

    for f in range(1, n_frames):
        dh = min_dist_h[f]
        da = min_dist_a[f]
        bs = ball_speed[f]

        if not in_flight:
            if bs >= 3.5 and min(dh, da) > 1.2:
                prev_dh = np.min(min_dist_h[max(0, f - 10):f])
                prev_da = np.min(min_dist_a[max(0, f - 10):f])
                if prev_dh < 1.8 and prev_dh < prev_da:
                    striker_team = 0
                    in_flight = True
                    flight_start = f
                elif prev_da < 1.8 and prev_da < prev_dh:
                    striker_team = 1
                    in_flight = True
                    flight_start = f
        else:
            if min(dh, da) < 1.5 or bs < 1.5 or out_of_bounds[f] or (f - flight_start > 75):
                in_flight = False
                receiver_team = 0 if dh < da else 1
                if striker_team == receiver_team:
                    pass_team[flight_start:f] = striker_team
                striker_team = -1

    print("Step 3: Dual-criterion touchline boundary & stoppage dead ball detection...")
    df_speed = pd.Series(mean_speed).interpolate().bfill().ffill()
    smooth_speed = df_speed.rolling(50, min_periods=1, center=True).mean().values

    dead_mask = np.zeros(n_frames, dtype=bool)

    # 3.1 Physical out-of-bounds stoppage
    in_oob = False
    start_oob = 0
    for f in range(n_frames):
        if out_of_bounds[f]:
            if not in_oob:
                in_oob = True
                start_oob = f
        else:
            if in_oob:
                in_oob = False
                if f - start_oob >= oob_min_len:
                    dead_mask[start_oob:f] = True
    if in_oob and (n_frames - start_oob >= oob_min_len):
        dead_mask[start_oob:n_frames] = True

    # 3.2 Open-field collective deceleration stoppage
    decel_candidate = (smooth_speed < speed_thresh) & (np.minimum(min_dist_h, min_dist_a) > 2.0)
    in_decel = False
    start_decel = 0
    for f in range(n_frames):
        if decel_candidate[f]:
            if not in_decel:
                in_decel = True
                start_decel = f
        else:
            if in_decel:
                in_decel = False
                if f - start_decel >= speed_min_len:
                    dead_mask[start_decel:f] = True
    if in_decel and (n_frames - start_decel >= speed_min_len):
        dead_mask[start_decel:n_frames] = True

    print("Step 4: Controlled possession state machine with turnover hysteresis...")
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
            # Check if this frame is a locked pass in flight between teammates
            locked_team = pass_team[f]
            if locked_team != -1:
                state = locked_team
                if locked_team == 0:
                    h_control_count = 5
                    a_control_count = 0
                else:
                    a_control_count = 5
                    h_control_count = 0
                labels_out[f] = "H" if state == HOME else "A"
                continue

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
                # Stationary ball protection: requires kick impulse or moving player control
                is_kicked = (bs >= restart_min_speed)
                is_moving_ctrl = (np.minimum(dh, da) < 1.5) and (smooth_speed[f] >= 1.45)
                if in_field and (is_kicked or is_moving_ctrl):
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

    print("Step 5: Glitch filtering...")
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
