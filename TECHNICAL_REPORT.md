# More than a Hack 2027 — Phase 1: Controlled Possession Pipeline Technical Report

**Team:** E-GEN  
**Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), LaLiga EA Sports 2024-25 Matchday 2  
**Target:** Continuous frame-by-frame ball possession classification (`H`, `A`, `D`, `X`) for 199,319 tactical video frames (25 fps).

---

## 1. Approach Overview

Standard nearest-player algorithms with symmetric rolling-window majority votes suffer from fundamental physics flaws on continuous tracking data:
1. **The "Pass-in-Flight" Dilemma:** When Team H executes a pass across the midfield, the ball travels through contested space. If an opponent (Team A) is closer to the ball flight vector, naive proximity heuristics flip possession to Team A, splitting real continuous possessions into fragmented pieces and destroying `spellF1`.
2. **Dead Ball Blindness & Missing Ball Stoppages:** Tracking systems track players and ball even when play has stopped. An algorithm that only checks whether the ball is tracked fails to detect when the ball crossed out of bounds. Furthermore, when the ball goes out of play or is missing during set-piece setups, naive pipelines guess randomly. When a player walks toward a stationary ball to take a restart, naive heuristics falsely assume play has restarted.
3. **Contested Deflections & Tackle Rebounds:** During an established attack, defending players frequently challenge, poke, or deflect the ball without winning control. In football analytics, a tackle or deflection where the ball is immediately regained by the attacking team does NOT constitute a turnover. Naive thresholding creates brief spurious 1-2 second turnovers that ruin spell IoU.

Our **Enhanced Controlled Possession Pipeline** solves these challenges through a physics-informed state machine architecture:
- **Kinematic Ball Trajectory Imputation & Velocity Modeling:** Short ball occlusions ($\le 30$ frames / 1.2s) during live play are interpolated using physical velocity bounds ($\le 35$ m/s), eliminating artificial dropouts while accurately measuring instantaneous ball velocity vectors $\vec{v}_{ball}$.
- **Pass-in-Flight & Teammate Reception Tracking:** When the ball launches with velocity $v_{ball} \ge 3.5$ m/s from a player of Team $k$ and arrives at a teammate of Team $k$, all intermediate transit frames are explicitly locked to Team $k$. Opponents standing near the pass trajectory are prohibited from triggering false turnovers, preserving whole-spell temporal integrity.
- **Multi-Criterion Dead Ball (`D`) Detection:**
  - *Physical boundary check:* Coordinate projection against pitch dimensions ($105.3 \times 68.0$ m). When the ball travels outside pitch margins ($x < -0.2$, $x > 105.2$, $y < -0.2$, $y > 68.2$) for $\ge 25$ frames, it is marked as an out-of-bounds stoppage.
  - *Missing-Ball Stoppage Detection:* The ball being untracked (NaN) during low-to-moderate collective player movement ($\bar{v}_{players} < 1.85$ m/s for $\ge 90$ frames) indicates a ball retrieval or set-piece placement (corner, throw-in, goal kick).
  - *Collective deceleration:* When play is stopped (fouls, injuries, referee whistles), the rolling average speed of all 22 tracked players collapses from live match pace ($\sim 2.0\text{--}2.5$ m/s) down to a walk ($< 1.20$ m/s). Requiring sustained low velocity ($\ge 50$ frames) flags true match stoppages while preserving slow tactical build-up.
  - *Stationary Ball Restart Protection:* While in dead ball (`D`), players walking near a stationary ball cannot exit `D` until the ball is actively kicked ($v_{ball} \ge 2.8$ m/s) into play.
- **Turnover Hysteresis & Inertia State Machine:**
  - *Inertia rule:* While the ball is in flight, contested, or loose on the pitch, possession remains with the team that previously controlled it (adhering strictly to official challenge rules).
  - *Turnover threshold:* An opposing team only gains possession when one of their players establishes controlled contact ($d < 2.8$ m for $\ge 10$ consecutive frames).
- **Contested Rebound Absorption & Temporal Smoothing:**
  - Absorbs brief turnover attempts ($< 50$ frames / 2.0s from defensive deflections or tackles) back into the surrounding possession spell, preventing the fragmentation of long possession sequences.
  - Cleans micro-stoppage glitches ($< 35$ frames).

---

## 2. Match Assets Used

| Asset | Utilized | Purpose |
| :--- | :---: | :--- |
| `BAR-ATH_tactical_atd.txt` | **Yes** | 25 fps raw tracking of 46 tracklets and ball coordinates. |
| `BAR-ATH_tactical_atd.xml` | **Yes** | FIFA/EPTS metadata assigning track slots to Barcelona (`H`) and Athletic Club (`A`). |
| `BAR-ATH_tactical_smart_tagging.xml` | **Evaluated** | Analyzed for tactical event alignment and set-piece validation. |
| Video (`.mp4`) | **No** | Full algorithmic inference is performed directly from tracking geometry, ensuring extreme reproducibility and running in ~4 seconds. |

---

## 3. Training & Validation Data

- **External / Hand Labels on Provided Match:** **None** (100% zero-shot / algorithmic inference, strictly conforming to Rule 1 and Rule 2 of `RULES.md`).
- **Validation:** Validated against the public 5-minute ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`) using the official scorer (`evaluation/run_evaluation.py`).

---

## 4. Evaluation Results

Scored on the official ground truth:

| Model | spellF1 (Ranking Metric) | macroF1 (1st Tie-break) | Possessions (GT = 17) | Mean Duration (GT = 14.2s) | spellF1 @ 0.5 IoU | Transition F1 | Frag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`submission_controlled_possession` (Ours)** | **0.5176** | **0.8065** | **17** (+0) | **15.6s** | **0.6471** (11/17 matched) | **0.2927** | **1.05** |
| Previous Option A Baseline | 0.4821 | 0.7368 | 22 (+5) | 12.5s | 0.6154 (12/17 matched) | 0.2174 | 1.29 |
| `baseline_smooth125` | 0.4242 | 0.5812 | 16 (-1) | 18.5s | 0.4242 (7/17 matched) | 0.1667 | 0.81 |
| `baseline_smooth50` (Official Floor) | 0.3256 | 0.6207 | 26 (+9) | 10.8s | 0.3256 (7/17 matched) | 0.0816 | 1.43 |
| `baseline_smooth25` | 0.2868 | 0.6200 | 36 (+19) | 7.6s | 0.3019 (8/17 matched) | 0.2034 | 1.90 |
| `baseline_raw` | 0.0378 | 0.5972 | 131 (+114) | 2.0s | 0.0270 (2/17 matched) | 0.1410 | 6.52 |

### Key Performance Highlights
- **+58.9% relative gain on `spellF1`** over the official benchmark floor (`baseline_smooth50`: 0.3256 $\rightarrow$ **0.5176**).
- **Macro-F1 breaks 80% (0.8065)**, with balanced accuracy across all three classes:
  - **Barcelona (`H`): F1 = 0.8606** (Precision: 84.5%, Recall: 87.6%)
  - **Athletic Club (`A`): F1 = 0.8436** (Precision: 78.0%, Recall: 91.9%)
  - **Dead Ball (`D`): F1 = 0.7154** (Precision: 95.4%, Recall: 57.3%)
- **Exact Possession Spell Count:** Correctly predicts **17 possessions** (exact match to ground truth 17).
- **Mean Possession Duration:** **15.6s** vs GT 14.2s (only +1.4s difference, compared to baselines deviating by -6.6s to +4.3s).
- **Alive/Dead Discrimination:** **F1 = 0.9471**, team accuracy while alive = **90.1%**.
- **Fragmentation Ratio:** **1.05** (essentially identical to ground truth 1.00).

---

## 5. Submission Verification

The submission file `evaluation/results/submission_controlled_possession.csv` passed all checks executed by `evaluation/validate_submission.py`:
- Header: exactly `frame,label`
- Total rows: exactly 199,319
- Monotonic frames: `0..199318` without missing or duplicate frames
- Labels: strictly in `{'H', 'A', 'D', 'X'}` with 0 blank cells
- Reported possessions: 397 across the full match.

---

## 6. How to Reproduce

In the repository root:
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the pipeline (generates submission, validates, and runs scorer)
python possession_pipeline.py
```
Execution finishes completely in **~4 seconds**.
