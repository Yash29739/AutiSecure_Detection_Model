import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from config.config import TRAIT_COLUMNS, MODEL_PATH


def train_and_save_model(train_path="train_dataset.csv", test_path="test_dataset.csv"):
    """
    Trains a Random Forest model on the training dataset and evaluates on a separate test dataset.
    Automatically aligns feature columns between train and test sets.
    """
    print("Loading training and testing datasets...")

    # --- Load datasets ---
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"✅ Training data shape: {train_df.shape}")
    print(f"✅ Testing data shape: {test_df.shape}")

    # --- Drop non-feature columns ---
    for df in [train_df, test_df]:
        if "Video_ID" in df.columns:
            df.drop(columns=["Video_ID"], inplace=True)
        df.dropna(subset=["is_autistic"], inplace=True)

    # --- Split into features and target ---
    X_train = train_df.drop(columns=["is_autistic"])
    y_train = train_df["is_autistic"]
    X_test = test_df.drop(columns=["is_autistic"])
    y_test = test_df["is_autistic"]

    # --- Align feature columns between train and test ---
    # Find common columns (shared features)
    common_features = X_train.columns.intersection(X_test.columns).tolist()
    X_train = X_train[common_features]
    X_test = X_test[common_features]

    print(f"✅ Using {len(common_features)} shared features for training/testing.")
    if len(common_features) != X_train.shape[1]:
        print("⚠️ Some features were dropped because they were not present in both datasets.")

    # --- Train model ---
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # --- Evaluate model ---
    print("Evaluating model on test dataset...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    # --- Print metrics summary ---
    print("\n✅ Model training and evaluation complete.")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")

    # --- Save model and metadata ---
    joblib.dump({
        "model": model,
        "metrics": metrics,
        "features": common_features  # Save the features used for reference
    }, MODEL_PATH)

    print(f"\n💾 Model and metrics saved to '{MODEL_PATH}'")
    return model, metrics

def load_model():
    """Loads a pre-trained model and its metrics."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from '{MODEL_PATH}'...")
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict) and "model" in data:
            return data["model"], data.get("metrics", {})
        else:
            return data, {}
    else:
        raise FileNotFoundError("Model not found. Please train a model first.")


def generate_prediction_and_report(features, model_data=None):
    """
    Generates a detailed prediction report for a single patient/sample.
    Automatically handles both (model, metrics) tuple or direct load from disk.
    """

    print("\n--- Generating Patient Report ---")

    # --- Load model and metadata properly ---
    if model_data is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file not found. Please train a model first.")
        data = joblib.load(MODEL_PATH)
        model = data["model"]
        metrics = data.get("metrics", {})
        model_features = data.get("features", [])
    else:
        # Handle both tuple (model, metrics) and dict from joblib
        if isinstance(model_data, tuple):
            model, metrics = model_data
            # Load feature info from saved model
            saved = joblib.load(MODEL_PATH)
            model_features = saved.get("features", [])
        elif isinstance(model_data, dict):
            model = model_data["model"]
            metrics = model_data.get("metrics", {})
            model_features = model_data.get("features", [])
        else:
            raise TypeError("Invalid model_data format. Expected tuple or dict.")

    # --- Convert features to DataFrame ---
    input_df = pd.DataFrame([features])

    # --- Align input with model feature set ---
    missing_features = [f for f in model_features if f not in input_df.columns]
    for f in missing_features:
        input_df[f] = 0  # default fill for missing traits

    input_df = input_df[model_features]

    # --- Make prediction ---
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0]

    # --- Build report ---
    predicted_label = "AUTISTIC" if prediction[0] == 1 else "NOT AUTISTIC"
    confidence = prediction_proba[1] if prediction[0] == 1 else prediction_proba[0]

    report = {
        "detected_traits": features,
        "final_prediction": {
            "label": predicted_label,
            "confidence": f"{confidence:.2%}",
            "likelihood_score": round(float(confidence), 4)
        },
        "model_info": {
            "used_features": model_features,
            "performance": metrics if metrics else "No metrics available"
        }
    }

    print(f"✅ Prediction: {predicted_label} (Confidence: {confidence:.2%})")
    return report

