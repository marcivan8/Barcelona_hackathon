# More than a Hack 2027 — Phase 1: Controlled Possession Pipeline Technical Report

**Team:** Autonomous Football Analytics  
**Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), LaLiga EA Sports 2024-25 Matchday 2  
**Target:** Continuous frame-by-frame ball possession classification (`H`, `A`, `D`, `X`) for 199,319 tactical video frames.

---

## 1. Approach Overview

Standard nearest-player algorithms with symmetric rolling-window majority votes suffer from fundamental flaws on continuous tracking data:
1. **Pass / In-flight ball flickering:** When a ball is passed from player 1 to player 2 of Team H, it may momentarily pass within 3 meters of an opponent (Team A). Symmetric window filters switch to Team A, splitting real continuous possessions into fragmented pieces and destroying `spellF1`.
2. **Dead Ball Blindness:** The tracking system tracks players and ball even when play has stopped. An algorithm that only checks whether the ball is tracked fails to detect when the ball crossed out of bounds or when the match is in a dead-ball stoppage.
3. **Occlusions & Missing Ball:** The ball is untracked on ~41% of graded frames, leading naive baselines to guess `D` or invent spurious state handovers.

Our **Controlled Possession Pipeline** introduces a physics-informed state machine architecture:
- **Kinematic Ball Trajectory Imputation & Velocity Modeling:** Short ball occlusions ($\le 30$ frames / 1.2s) during live play are interpolated using physical velocity bounds ($\le 35$ m/s), eliminating artificial dropouts while accurately measuring instantaneous ball velocity.
- **Physical Boundary & Collective Deceleration Dead Ball (`D`) Detector:**
  - *Pitch boundary enforcement:* Exact coordinate projection against pitch dimensions ($105.3 \times 68.0$ m). When the ball travels outside pitch margins ($x < -0.2$, $x > 105.5$, $y < -0.2$, $y > 68.2$), it is marked out of play.
  - *Collective deceleration:* When play is stopped (fouls, injuries, referee whistles), the rolling average speed of all 22 tracked players collapses from live match pace ($\sim 2.0\text{--}2.5$ m/s) down to a walk ($< 1.20$ m/s). Requiring sustained low velocity ($\ge 50$ frames) flags true match stoppages while preserving slow tactical build-up.
- **Turnover Hysteresis & Inertia State Machine:**
  - *Inertia rule:* While the ball is in flight, contested, or loose on the pitch, possession remains with the team that previously controlled it (adhering strictly to the official challenge definition).
  - *Turnover threshold:* An opposing team only gains possession when one of their players establishes controlled contact ($d < 2.5$ m for $\ge 12$ consecutive frames).
  - *Restart gating:* Exiting a dead ball state requires the ball to be in-bounds and actively kicked ($v_{ball} > 2.5$ m/s) or brought under close proximity control.
- **Micro-Spell Glitch Filter:** Residual single-frame deflections ($< 25$ frames / 1.0s) are merged back into the surrounding spell.

---

## 2. Match Assets Used

| Asset | Utilized | Purpose |
| :--- | :---: | :--- |
| `BAR-ATH_tactical_atd.txt` | **Yes** | 25 fps raw tracking of 46 tracklets and ball coordinates. |
| `BAR-ATH_tactical_atd.xml` | **Yes** | FIFA/EPTS metadata assigning track slots to Barcelona (`H`) and Athletic Club (`A`). |
| `BAR-ATH_tactical_smart_tagging.xml` | **Analyzed** | Evaluated set piece restart markers; the physical tracking features proved higher temporal precision for restart instants. |
| `BAR-ATH_tactical_homography.csv` | **Optional** | Pitch geometry verification. |
| Video (`.mp4`) | **No** | Full algorithmic inference is performed directly from tracking geometry, ensuring extreme reproducibility and running in under 4 seconds. |

---

## 3. Training & Validation Data

- **External / Hand Labels on Provided Match:** **None** (100% zero-shot / algorithmic inference, strictly conforming to Rule 1 and Rule 2 of `RULES.md`).
- **Validation:** Validated against the public 5-minute ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`) using the official scorer (`evaluation/run_evaluation.py`).

---

## 4. Evaluation Results

Scored on the official ground truth:

| Model | macroF1 | spellF1 (Ranking Metric) | Possessions (GT = 17) | Mean Duration (GT = 14.2s) | Transition F1 | Share Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`submission_controlled_possession` (Ours)** | **0.7499** | **0.4615** | **22** | **12.3s** | **0.2609** | **0.0072** |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5s | 0.1667 | 0.0520 |
| `baseline_smooth50` (Official Floor) | 0.6207 | 0.3256 | 26 | 10.8s | 0.0816 | 0.0485 |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 | 7.6s | 0.2034 | 0.0556 |
| `baseline_raw` | 0.5972 | 0.0378 | 131 | 2.0s | 0.1410 | 0.0457 |

### Key Improvements
- **+41.7% relative improvement on `spellF1`** over the official baseline floor (`baseline_smooth50`).
- **+12.9 points jump in `macroF1`** (0.7499 vs 0.6207).
- **Dead Ball Precision: 93.8%** (virtually zero false dead-ball interruptions during live possession).
- **Alive/Dead F1: 0.9342**, Team Accuracy: **87.6%**.
- **Mean Transition Latency:** -2.2 frames (under 90 milliseconds).

---

## 5. Submission Verification

The submission file `evaluation/results/submission_controlled_possession.csv` passed all checks executed by `evaluation/validate_submission.py`:
- Header: exactly `frame,label`
- Total rows: exactly 199,319
- Monotonic frames: `0..199318` without missing or duplicate frames
- Labels: strictly in `{'H', 'A', 'D', 'X'}` with 0 blank cells
- Reported possessions: 496 across the full 106 minutes of match play.

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

