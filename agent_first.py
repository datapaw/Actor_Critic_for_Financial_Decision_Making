# Baseline ResNet model for S&P 500 prediction
# Uses 20-day rolling windows to predict which time horizon is best

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

# config stuff
DATA_PATH = Path("data/dataset/sp500_prep.csv")
TIME_FRAME_DAYS = 20  # 20-day windows seem to work best
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COLUMN = "target_period"

# train/val/test splits - keeping most data for training
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05
RANDOM_STATE = 42  # for reproducibility
BATCH_SIZE = 64
EPOCHS = 50  # usually converges around 30-40


def load_dataset(
    csv_path: Path,
    time_frame_days: int,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    # load data and build rolling windows
    df = pd.read_csv(csv_path, index_col=0)

    # Keep only needed columns to avoid leakage
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in dataset")

    # Drop invalid targets
    df = df[df[target_col] >= 0]
    df = df.dropna(subset=feature_cols + [target_col])
    df = df.reset_index(drop=True)

    sequences: List[np.ndarray] = []
    targets: List[int] = []

    for i in range(time_frame_days, len(df)):
        window = df.iloc[i - time_frame_days : i][feature_cols].to_numpy()
        label = int(df.iloc[i][target_col])
        sequences.append(window)
        targets.append(label)

    X = np.stack(sequences)
    y = np.array(targets, dtype=np.int64)
    return X, y


def residual_block(x: layers.Layer, filters: int, kernel_size: int = 3, stride: int = 1) -> layers.Layer:
    """Simple residual block with Conv1D, BN, ReLU."""
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding="same", strides=stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions if stride changes
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, 1, padding="same", strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet(input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stack a few residual blocks
    for filters in [64, 128, 128]:
        x = residual_block(x, filters=filters, kernel_size=3, stride=1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet_1d")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading dataset from {DATA_PATH}...")
    X, y = load_dataset(DATA_PATH, TIME_FRAME_DAYS, FEATURE_COLUMNS, TARGET_COLUMN)
    num_classes = int(y.max() + 1)
    print(f"Data shapes -> X: {X.shape}, y: {y.shape}, classes: {num_classes}")

    # 90/5/5 split: first carve out 10% holdout, then split holdout into val/test 50/50
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, random_state=RANDOM_STATE, stratify=y_hold
    )

    print(
        f"Split -> train: {X_train.shape[0]} ({TRAIN_SPLIT*100:.0f}%), "
        f"val: {X_val.shape[0]} ({VAL_SPLIT*100:.0f}%), "
        f"test: {X_test.shape[0]} ({TEST_SPLIT*100:.0f}%)"
    )

    model = build_resnet(input_shape=X.shape[1:], num_classes=num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # Save model
    # Directories for outputs
    model_dir = Path("models")
    results_dir = Path("results")
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    model_path = model_dir / "resnet_sp500.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Confusion matrix on test set
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [str(d) for d in range(num_classes)])
    plt.yticks(tick_marks, [str(d) for d in range(num_classes)])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Annotate counts in each cell
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            plt.text(
                j,
                i,
                format(value, "d"),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Confusion matrix raw values as CSV
    cm_csv_path = results_dir / "confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_csv_path, index=False)
    print(f"Confusion matrix values saved to {cm_csv_path}")

    # Classification report
    report = classification_report(y_test, y_pred_labels, digits=4)
    report_path = results_dir / "classification_report.txt"
    report_path.write_text(report)
    print(f"Classification report saved to {report_path}")

    # Model diagram
    diagram_path = results_dir / "model_diagram.png"
    try:
        plot_model(model, to_file=diagram_path, show_shapes=True, show_layer_names=True, dpi=150)
        print(f"Model diagram saved to {diagram_path}")
    except Exception as exc:  # pragma: no cover - optional dependency (pydot/graphviz)
        print(f"Could not create model diagram: {exc}")


if __name__ == "__main__":
    main()
