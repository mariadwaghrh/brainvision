import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci

mne.set_log_level("ERROR")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project folder, works anywhere
MNE_DATA_DIR = os.path.join(BASE_DIR, "mne_data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SUBJECTS = [1, 2]                 # PhysioNet EEG Motor Movement/Imagery volunteers
RUNS = [3, 4, 7, 8, 11, 12]       # left vs right fist runs (real + imagined)
WINDOW = (0.5, 2.5)               # seconds AFTER the cue, when the mu/beta drop happens
VLIM = (-2.5, 2.5)                # one fixed colour scale for every image (in z-score units)

for folder in [MNE_DATA_DIR, os.path.join(DATASET_DIR, "left_hand"), os.path.join(DATASET_DIR, "right_hand")]:
    os.makedirs(folder, exist_ok=True)
for old in glob.glob(os.path.join(DATASET_DIR, "*", "*.png")):
    os.remove(old)

print(f"[1/3] Loading PhysioNet EEG for subjects {SUBJECTS} into {MNE_DATA_DIR} ...")
powers, labels, info = [], [], None
for subject in SUBJECTS:
    fnames = eegbci.load_data(subject, RUNS, path=MNE_DATA_DIR, update_path=False)
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in fnames])
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

    # Motor rhythms: mu (8-13 Hz) and beta (13-30 Hz)
    raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    epochs = mne.Epochs(raw, events, dict(left=1, right=2), tmin=WINDOW[0], tmax=WINDOW[1],
                        picks="eeg", baseline=None, preload=True)
    info = epochs.info

    # Band power per channel, then z-scored per channel within this subject.
    # This removes each person's "normal" power map (no labels used), so what is
    # left is how this trial differs from the subject's average: the left/right signal.
    log_power = np.log(np.var(epochs.get_data(copy=True), axis=2))
    z = (log_power - log_power.mean(axis=0)) / log_power.std(axis=0)

    powers.append(z)
    labels.append(epochs.events[:, 2])

powers, labels = np.concatenate(powers), np.concatenate(labels)

print(f"Generating {len(labels)} brain topomap images...")
counts = {"left_hand": 0, "right_hand": 0}
for p, label_id in zip(powers, labels):
    label_name = "left_hand" if label_id == 1 else "right_hand"
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    mne.viz.plot_topomap(p, info, axes=ax, show=False, cmap="jet", contours=0,
                         vlim=VLIM, sensors=False)
    fig.savefig(os.path.join(DATASET_DIR, label_name, f"brain_{counts[label_name]:03d}.png"),
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    counts[label_name] += 1

print(f"Done! Images saved in {DATASET_DIR} (Left: {counts['left_hand']}, Right: {counts['right_hand']})")
