# Gaze Estimation — Progress Report

## 1. Motivation & Approach

Training a gaze estimation model from scratch with a small number of samples is not feasible — the model would not generalise to unseen users or environments. Instead, I chose to leverage an existing pre-trained model from [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation), which has been trained and validated on a large-scale dataset.

**Dataset:** 197,588 frames from 238 subjects with 3D gaze annotations — this provides strong generalisation across head poses, lighting conditions, and individual differences. Using this as a starting point gives us a functional and reliable gaze estimation backbone without needing to collect and annotate our own training data.

---

## 2. Model Output & The Calibration Problem

The model outputs **(pitch, yaw)** angles — the predicted gaze direction in 3D space — rather than screen-space **(x, y)** coordinates. This is a fundamental mismatch with our end goal of knowing where on the screen the user is looking.

To bridge this gap, I designed a **polynomial calibration layer** that learns the mapping:

```
f(pitch, yaw) → (screen_x, screen_y)
```

**Key design decisions:**

- **Polynomial mapping instead of a neural network.**
  The relationship between (pitch, yaw) and (screen_x, screen_y) is smooth and well-behaved — a degree-2 polynomial with cross terms (e.g. pitch², yaw², pitch×yaw) is expressive enough to capture the curvature introduced by screen geometry and individual head positioning. A more complex model would overfit on only 25 calibration samples.

- **Ridge regression for numerical stability.**
  With 25 data points and 6 polynomial features, the system is overdetermined but the feature matrix can still be ill-conditioned (e.g. if the user's yaw range is narrow). Ridge regression adds L2 regularisation (α = 1.0), which prevents the coefficients from blowing up and keeps predictions stable outside the calibration grid.

- **Separate regressors for x and y.**
  Screen x and screen y are fitted independently — one Ridge regressor maps (pitch, yaw) → screen_x, another maps (pitch, yaw) → screen_y. This keeps the problem simple and makes it easy to diagnose axis-specific errors (as seen in our results: x RMSE = 38px, y RMSE = 87px).

- **25-point grid covers the full gaze space.**
  The 5×5 grid is distributed across the screen with an 8% margin, so the calibration samples span the user's full angular range. With 30 frames averaged per point, each sample represents a stable mean gaze angle rather than a noisy single frame. The total session takes roughly 2–3 minutes.

- **No model retraining required.**
  The pre-trained gaze model's weights are frozen throughout. Calibration only fits the lightweight polynomial layer on top, meaning a new user can self-calibrate in a single session without any GPU training or data collection infrastructure.

- **Model-agnostic design.**
  Because the calibration layer only consumes (pitch, yaw) outputs, it is entirely decoupled from the backbone. If the underlying gaze model is replaced or improved, the same calibration procedure applies without any changes to the pipeline.

**Current calibration result:**
```
Calibration RMSE:  x = 38.0 px,  y = 87.2 px
```

The x-axis accuracy is reasonable. The y-axis error is higher, which is a known characteristic of gaze models — pitch (vertical gaze) is inherently harder to predict than yaw (horizontal) due to eyelid occlusion. Overall, the system can follow and predict estimated gaze position with acceptable accuracy for an initial prototype.

---

## 3. Current Pipeline Summary

```
Camera feed
    └── Face detection (RetinaFace)
        └── Face crop
            └── Gaze model (ResNet-34) → (pitch, yaw)
                └── Polynomial calibration → (screen_x, screen_y)
```

The full pipeline runs locally in real time. A calibration demo (`calibration_demo.py`) handles:
- **Phase 0**: Face preview — user confirms face is detected before starting
- **Phase 1**: 25-point grid calibration with fixation task (rotating random letters on each dot to maintain attention)
- **Phase 2**: Polynomial fit on collected (pitch, yaw) → (x, y) pairs
- **Phase 3**: Live gaze cursor with smoothing

---

## 4. Thoughts on Next Steps

Now that the full pipeline is functional and validated end-to-end, there are two parallel directions to consider:

### Direction A — Model Architecture Improvement
The current backbone is **ResNet-34**. With the dataset and pipeline already in place, architectural improvements (e.g., a more efficient or accurate backbone, attention mechanisms, or personalisation layers) could meaningfully reduce the RMSE — especially on the y-axis. This is a research-oriented track that could run continuously in the background.

### Direction B — App Deployment (PseudoTok)
For the summer milestone, I propose establishing a **cutoff point**: by a defined date, shift primary focus from model refinement to **deploying a working app on mobile (PseudoTok)**. The goal is to have a functional, testable app in hand for initial trials with anorexia-related settings.

### Proposed Parallel Strategy

| Track | Focus | Timeline |
|---|---|---|
| Deployment | Port to mobile app, integrate current model | → Summer cutoff |
| Research | Architecture refinement, improve RMSE | Ongoing in parallel |

The key advantage of this design: the calibration layer is **model-agnostic**. If a higher-performance model is trained later, it can be seamlessly swapped in without changing any other part of the pipeline.

---

## 5. Open Question for Discussion

> After the summer cutoff, should the priority shift fully to deployment and initial user trials, with model refinement becoming a secondary track? Or is there a minimum accuracy threshold the model must meet before it is suitable for use in anorexia-related experimental settings?
