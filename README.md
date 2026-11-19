# BrainVision: reading hand-movement intent from brain maps

An interactive, bilingual (English / Arabic) explainer that turns EEG recordings into images, then uses **computer vision** and **pattern recognition** to guess whether a person is about to move their **left** or **right** hand.

**Live site:** https://mariadwaghrh.github.io/brainvision/ *(after GitHub Pages is enabled)*

## What's inside

| Chapter | What you can do |
|---|---|
| Story | Scroll through the pipeline. A sticky panel follows one real recording from EEG signal to prediction. |
| Vision Lab | Pick any of the 180 brain maps and run every OpenCV step live in the browser: ROI mask, HSV channels, Canny, HOG, hemisphere histograms, and an inverted *jet* colormap that reveals the hidden activity. |
| Pattern Lab | Explore the PCA space, inspect SVM and Random Forest predictions, compare a single split with 5-fold cross-validation, and train your own k-NN live. |
| Play | Guess left or right, and compare your score with the SVM. |
| Version 1 → 2 | Why the first version failed and what fixed it. |

## Pipeline

1. **EEG** (`1_generate_topomaps.py`): PhysioNet EEG Motor Movement/Imagery, subjects 1–2, band-passed to 8–30 Hz (mu and beta rhythms). For the window 0.5–2.5 s after each cue, every electrode's log power is z-scored within its subject. Each trial is then drawn as a topomap with one fixed colour scale: 92 left, 88 right.
2. **Computer vision** (`2_cv_feature_extraction.py`): resize to 128×128, circular ROI mask, then three feature groups:
   - HSV hue/saturation histograms for each hemisphere and their difference (192)
   - colormap inversion, which turns each *jet* colour back into an activity value, shrunk to a 16×16 grid (256)
   - mirror asymmetry: the left half minus the mirrored right half (128)

   That gives 576 features per image.
3. **Pattern recognition** (`3_pr_classifier_demo.py`): StandardScaler → PCA (20 components) → SVM (RBF) and Random Forest. Accuracy is reported with 5-fold cross-validation, with the scaler and PCA refit inside each fold.
4. **Website export** (`website/build_data.py`): PCA coordinates, predictions, 5-fold scores and the images, written into `website/data.js`.

## Results

| Version | What it used | SVM, 5-fold CV | Random Forest, 5-fold CV |
|---|---|---|---|
| 1 | window around the cue, per-image colour scale, HSV + HOG | 48.9% | 46.7% |
| **2** | window after the cue, fixed colour scale, HSV + activity grid + mirror asymmetry | **67.2%** (±8.1) | **61.7%** (±9.8) |

Version 1 looked fine on one train/test split (64.4%), but cross-validation showed it was guessing. Every topomap was the same dark blue, and HOG mostly described the head outline. Version 2 fixed the images first, then replaced HOG with features that measure where the activity is.

For reference, CSP + logistic regression on the same raw EEG (not images) reaches about 73% with 5-fold CV, so the image pipeline keeps most of the available signal. Adding more volunteers (10 subjects) lowered signal-level accuracy to about 60%, because brain patterns differ a lot between people.

**Limits:**
- The data is small: 180 trials from 2 people, so scores move by about ±8% between folds.
- The per-subject z-score uses all of that subject's trials. No labels are involved, but test trials do contribute to the average.

## Run it

```bash
pip install -r requirements.txt
python 1_generate_topomaps.py      # downloads ~30 MB of EEG, writes dataset/
python 2_cv_feature_extraction.py  # writes cv_features.npz
python 3_pr_classifier_demo.py     # trains the models, writes the results figure
python website/build_data.py       # writes website/data.js
```

On Windows you can run `run_all.bat` instead. Then open `website/index.html`. It works offline because the images are bundled in `data.js`. Add `?lang=ar` to the address for Arabic.

The browser demos are plain JavaScript and canvas, with no libraries. Canny, HOG, HSV, the colormap inversion and k-NN are all reimplemented in `website/app.js`.

## Data

Goldberger et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23), 2000. Schalk et al., *BCI2000: A General-Purpose Brain-Computer Interface System*, IEEE TBME 51(6), 2004.
