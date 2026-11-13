# More than a Hack 2027 — Phase 1: Controlled Possession Pipeline Technical Report

**Team:** Autonomous Football Analytics  
**Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), LaLiga EA Sports 2024-25 Matchday 2  
**Target:** Continuous frame-by-frame ball possession classification (`H`, `A`, `D`, `X`) for 199,319 tactical video frames.

---

## 1. Approach Overview

Standard nearest-player algorithms with symmetric rolling-window majority votes suffer from fundamental flaws on continuous tracking data:
1. **The "Pass-in-Flight" Dilemma:** When a ball is passed from player 1 to player 2 of Team H, it travels through contested space. If an opponent (Team A) stands near the trajectory, naive distance metrics switch possession to Team A, splitting real continuous possessions into fragmented pieces and destroying `spellF1`.
2. **Dead Ball Blindness & Stationary Ball Walking:** Tracking systems track players and ball even when play has stopped. An algorithm that only checks whether the ball is tracked fails to detect when the ball crossed out of bounds. Furthermore, when a player walks towards a stationary ball to take a goal kick or free kick, naive heuristics falsely assume play has restarted.
3. **Occlusions & Missing Ball:** The ball is untracked on ~41% of graded frames, leading naive baselines to guess `D` or invent spurious state handovers.

Our **Option A Enhanced Controlled Possession Pipeline** introduces a physics-informed state machine architecture:
- **Kinematic Ball Trajectory Imputation & Velocity Modeling:** Short ball occlusions ($\le 30$ frames / 1.2s) during live play are interpolated using physical velocity bounds ($\le 35$ m/s), eliminating artificial dropouts while accurately measuring instantaneous ball velocity vectors $\vec{v}_{ball}$.
- **Pass-in-Flight & Teammate Reception Tracking:** When the ball launches with velocity $v_{ball} \ge 3.5$ m/s from a player of Team $k$ and arrives at a teammate of Team $k$, all intermediate transit frames are explicitly locked to Team $k$. Opponents standing near the pass trajectory are prohibited from triggering false turnovers, preserving whole-spell temporal integrity.
- **Dual-Criterion Touchline Boundary & Stoppage Dead Ball (`D`) Detector:**
  - *Physical boundary check:* Coordinate projection against pitch dimensions ($105.3 \times 68.0$ m). When the ball travels outside pitch margins ($x < -0.2$, $x > 105.5$, $y < -0.2$, $y > 68.2$) for $\ge 25$ frames, it is marked as an out-of-bounds stoppage.
  - *Collective deceleration:* When play is stopped (fouls, injuries, referee whistles), the rolling average speed of all 22 tracked players collapses from live match pace ($\sim 2.0\text{--}2.5$ m/s) down to a walk ($< 1.20$ m/s). Requiring sustained low velocity ($\ge 50$ frames) flags true match stoppages while preserving slow tactical build-up.
  - *Stationary Ball Restart Protection:* While in dead ball (`D`), players walking near a stationary ball ($v_{ball} < 1.8$ m/s) cannot exit `D` until the ball is actively kicked ($v_{ball} \ge 2.5$ m/s) into play.
- **Turnover Hysteresis & Inertia State Machine:**
  - *Inertia rule:* While the ball is in flight, contested, or loose on the pitch, possession remains with the team that previously controlled it (adhering strictly to official challenge rules).
  - *Turnover threshold:* An opposing team only gains possession when one of their players establishes controlled contact ($d < 2.5$ m for $\ge 12$ consecutive frames).
- **Micro-Spell Glitch Filter:** Residual single-frame deflections ($< 25$ frames / 1.0s) are merged back into the surrounding spell.

---

## 2. Match Assets Used

| Asset | Utilized | Purpose |
| :--- | :---: | :--- |
| `BAR-ATH_tactical_atd.txt` | **Yes** | 25 fps raw tracking of 46 tracklets and ball coordinates. |
| `BAR-ATH_tactical_atd.xml` | **Yes** | FIFA/EPTS metadata assigning track slots to Barcelona (`H`) and Athletic Club (`A`). |
| `BAR-ATH_tactical_homography.csv` | **Optional** | Pitch geometry verification. |
| Video (`.mp4`) | **No** | Full algorithmic inference is performed directly from tracking geometry, ensuring extreme reproducibility and running in ~3 seconds. |

---

## 3. Training & Validation Data

- **External / Hand Labels on Provided Match:** **None** (100% zero-shot / algorithmic inference, strictly conforming to Rule 1 and Rule 2 of `RULES.md`).
- **Validation:** Validated against the public 5-minute ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`) using the official scorer (`evaluation/run_evaluation.py`).

---

## 4. Evaluation Results

Scored on the official ground truth:

| Model | spellF1 (Ranking Metric) | macroF1 (1st Tie-break) | Possessions (GT = 17) | Mean Duration (GT = 14.2s) | spellF1 @ 0.5 IoU | Transition F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`submission_controlled_possession` (Option A)** | **0.4821** | **0.7368** | **22** | **12.5s** | **0.6154** (12/17 matched) | **0.2174** |
| `baseline_smooth125` | 0.4242 | 0.5812 | 16 | 18.5s | 0.4242 (7/17 matched) | 0.1667 |
| `baseline_smooth50` (Official Floor) | 0.3256 | 0.6207 | 26 | 10.8s | 0.3256 (7/17 matched) | 0.0816 |
| `baseline_smooth25` | 0.2868 | 0.6200 | 36 | 7.6s | 0.3019 (8/17 matched) | 0.2034 |
| `baseline_raw` | 0.0378 | 0.5972 | 131 | 2.0s | 0.0270 (2/17 matched) | 0.1410 |

### Key Improvements
- **+48.1% relative improvement on `spellF1`** over the official baseline floor (`baseline_smooth50`).
- **70.6% recall at IoU 0.5** (recovers 12 out of 17 possessions as full spells vs. only 7 for the baselines).
- **Alive/Dead F1: 0.9287**, Team Accuracy: **88.7%**.
- **Dead Ball Precision: 93.5%**.

---

## 5. Submission Verification

The submission file `evaluation/results/submission_controlled_possession.csv` passed all checks executed by `evaluation/validate_submission.py`:
- Header: exactly `frame,label`
- Total rows: exactly 199,319
- Monotonic frames: `0..199318` without missing or duplicate frames
- Labels: strictly in `{'H', 'A', 'D', 'X'}` with 0 blank cells
- Reported possessions: 509 across the full 106 minutes of match play.

---

## 6. How to Reproduce

In the repository root:
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the pipeline (generates submission, validates, and runs scorer)
python possession_pipeline.py
```
Execution takes approximately **3 to 4 seconds**.
