import os
import glob
import json
import traceback
from pathlib import Path

import numpy as np
import joblib
import sklearn  # just to print version

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


LABELS = ["HELLO", "THANK_YOU", "YES", "CONFIRM"]


def project_root() -> Path:
    # training/train_static_model.py -> training -> project root
    return Path(__file__).resolve().parents[1]


def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # atomic replace


def safe_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic replace


def safe_joblib_dump(path: Path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, tmp)
    tmp.replace(path)  # atomic replace


def load_data(data_root: Path):
    X, y = [], []
    per_label_counts = {}

    for idx, lab in enumerate(LABELS):
        files = glob.glob(str(data_root / lab / "*.json"))
        per_label_counts[lab] = len(files)

        for fp in files:
            try:
                vec = json.loads(Path(fp).read_text(encoding="utf-8"))
                if not isinstance(vec, list):
                    continue
                if len(vec) != 128:
                    continue
                X.append(vec)
                y.append(idx)
            except Exception:
                # skip corrupt json
                continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, per_label_counts


def main():
    root = project_root()
    data_root = root / "training" / "data" / "static"
    models_dir = root / "models"
    model_path = models_dir / "static_lr.joblib"
    labelmap_path = models_dir / "label_map.json"
    log_path = models_dir / "train_log.txt"

    try:
        lines = []
        lines.append("=== TRAINING SCRIPT STARTED ===")
        lines.append(f"Project root: {root}")
        lines.append(f"Python executable: {os.sys.executable}")
        lines.append(f"scikit-learn: {sklearn.__version__}")
        lines.append(f"Data root: {data_root}")

        X, y, counts = load_data(data_root)
        lines.append("=== Data summary (files per label) ===")
        for k, v in counts.items():
            lines.append(f"{k}: {v} files")

        lines.append(f"Loaded X shape: {X.shape}")
        lines.append(f"Loaded y shape: {y.shape}")

        # Write log early so you always have it even if something fails later
        safe_write_text(log_path, "\n".join(lines) + "\n")

        if X.shape[0] == 0:
            raise RuntimeError(
                "No training samples loaded.\n"
                "Check you have JSON files in:\n"
                "  training/data/static/HELLO\n"
                "  training/data/static/THANK_YOU\n"
                "  training/data/static/YES\n"
                "Also ensure each JSON contains 128 numbers."
            )

        if X.shape[1] != 128:
            raise RuntimeError(f"Expected 128 features, got {X.shape[1]}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Model
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000))
        ])

        clf.fit(X_train, y_train)

        acc = float(clf.score(X_test, y_test))
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, target_names=LABELS)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Save label map FIRST (so it never stays empty)
        label_map = {str(i): lab for i, lab in enumerate(LABELS)}
        safe_write_json(labelmap_path, label_map)

        # Save model (atomic)
        safe_joblib_dump(model_path, clf)

        # Update log with results
        lines.append(f"Test accuracy: {acc:.4f}")
        lines.append("=== Classification report ===")
        lines.append(report)
        lines.append("=== Confusion matrix ===")
        lines.append(str(cm))
        lines.append(f"Saved label map: {labelmap_path} (bytes={labelmap_path.stat().st_size})")
        lines.append(f"Saved model: {model_path} (bytes={model_path.stat().st_size})")
        lines.append("=== TRAINING COMPLETE ===")

        safe_write_text(log_path, "\n".join(lines) + "\n")
        print("\n".join(lines))

    except Exception as e:
        # Always write full error + traceback to train_log.txt
        tb = traceback.format_exc()
        err_text = f"=== TRAINING FAILED ===\nError: {e}\n\nTraceback:\n{tb}\n"
        safe_write_text(log_path, err_text)
        print(err_text)
        raise


if __name__ == "__main__":
    main()
