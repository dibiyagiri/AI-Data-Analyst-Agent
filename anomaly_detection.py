
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df):
    """Detect unusual sales records using Isolation Forest."""

    data = df.copy()

    features = ["quantity", "price", "revenue"]

    # Ensure numeric values
    for col in features:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    # Exclude rows with missing or infinite values
    valid = data[features].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(valid) < 10:
        raise ValueError(
            "At least 10 valid records are required."
        )

    # Train the anomaly detection model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )

    predictions = model.fit_predict(valid)

    results = data.loc[valid.index].copy()

    results["anomaly"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    return results
