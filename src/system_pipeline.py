import joblib
import numpy as np
import pandas as pd
import logging
import threading
import time
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DriverSafetySystem:
    """
    Production-grade Explainable + Monitored Decision Intelligence System.
    Stateless inference.
    Thread-safe metrics.
    Interview-ready.
    """

    # ------------------------
    # INIT
    # ------------------------
    def __init__(self, model_path: str):

        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")

            if hasattr(self.model, "feature_names_in_"):
                self.MODEL_FEATURES = list(self.model.feature_names_in_)
                logger.info(f"Model features: {self.MODEL_FEATURES}")
            else:
                raise RuntimeError(
                    "Model missing feature_names_in_. Retrain with sklearn >=1.0"
                )

            # Extract feature importance safely
            model_step = None

            if hasattr(self.model, "named_steps"):
                model_step = self.model.named_steps.get("model")

            if model_step is None:
                model_step = self.model

            if hasattr(model_step, "feature_importances_"):
                self.feature_importances = model_step.feature_importances_
            else:
                self.feature_importances = None

            # Metadata
            self.model_version = "rf_v1.0"
            self.model_type = type(model_step).__name__

            # Monitoring state (thread-safe)
            self._metrics_lock = threading.Lock()
            self.total_requests = 0
            self.total_drowsy = 0
            self.total_latency = 0.0

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # ------------------------
    # HEALTH CHECK
    # ------------------------
    def health_check(self):

        if self.model is None:
            raise RuntimeError("Model not loaded")

        dummy_data = {feature: 0 for feature in self.MODEL_FEATURES}

        dummy_data.update({
            "Alertness": 1,
            "Seatbelt": 1,
            "HR": 70,
            "prev_alertness": 1
        })

        dummy = pd.DataFrame([dummy_data])[self.MODEL_FEATURES]
        self.model.predict(dummy)

    # ------------------------
    # MAIN ENTRY
    # ------------------------
    def analyze(self, input_data: Dict) -> Dict:

        start_time = time.time()

        self._validate_input(input_data)
        features = self._prepare_features(input_data)

        try:
            probabilities = self.model.predict_proba(features)[0]
        except Exception:
            raise RuntimeError("Model inference failure")

        pred_index = int(np.argmax(probabilities))

        ml_prediction = "Drowsy" if pred_index == 1 else "Alert"
        ml_confidence = float(probabilities[pred_index])
        ml_confidence = min(max(ml_confidence, 0.0), 1.0)

        # Stateless risk scoring
        risk_score, risk_factors = self._compute_risk_score(
            input_data,
            ml_confidence
        )

        risk_state = self._map_risk_state(risk_score)
        decision = self._decision_engine(risk_state)

        explanations = self._generate_explanations(
            input_data,
            ml_prediction,
            ml_confidence
        )

        top_features = self._get_top_contributors(features)

        latency = time.time() - start_time

        # Thread-safe metrics update
        with self._metrics_lock:
            self.total_requests += 1
            self.total_latency += latency
            if ml_prediction == "Drowsy":
                self.total_drowsy += 1

        return {
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "confidence_level": self._interpret_confidence(ml_confidence),
            "risk_score": risk_score,
            "risk_state": risk_state,
            "risk_factors": risk_factors,
            "decision": decision,
            "top_contributing_features": top_features,
            "explanations": explanations,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "inference_latency_ms": round(latency * 1000, 2)
        }

    # ------------------------
    # VALIDATION
    # ------------------------
    def _validate_input(self, data: Dict):

        required = set(self.MODEL_FEATURES)
        provided = set(data.keys())

        if not required.issubset(provided):
            raise ValueError("Input schema mismatch detected")

    # ------------------------
    # FEATURE PREP
    # ------------------------
    def _prepare_features(self, data: Dict) -> pd.DataFrame:
        return pd.DataFrame([data])[self.MODEL_FEATURES]

    # ------------------------
    # RISK ENGINE
    # ------------------------
    def _compute_risk_score(
        self,
        data: Dict,
        confidence: float
    ) -> Tuple[int, List[str]]:

        score = 0
        risk_factors = []

        if data["Fatigue"] > 6:
            score += 30
            risk_factors.append("High fatigue")

        if data["Alertness"] < 0.5:
            score += 25
            risk_factors.append("Low alertness")

        if data["HR"] > 100:
            score += 15
            risk_factors.append("Elevated heart rate")

        if confidence < 0.55:
            score += 25
            risk_factors.append("Model uncertainty")

        alert_drop = data["prev_alertness"] - data["Alertness"]

        if alert_drop > 0.25:
            score += 20
            risk_factors.append("Rapid alertness drop")

        if data["Speed"] == 0 and data["Fatigue"] > 7:
            score += 20
            risk_factors.append("Stationary but highly fatigued")

        return max(0, min(score, 100)), risk_factors

    # ------------------------
    # RISK STATE
    # ------------------------
    def _map_risk_state(self, score: int) -> str:

        if score >= 70:
            return "CRITICAL"
        elif score >= 40:
            return "MODERATE"
        return "LOW"

    # ------------------------
    # DECISION ENGINE
    # ------------------------
    def _decision_engine(self, state: str) -> Dict:

        decisions = {
            "CRITICAL": {
                "action": "Recommend immediate break",
                "severity": "HIGH",
                "message": "Risk increasing rapidly"
            },
            "MODERATE": {
                "action": "Suggest rest soon",
                "severity": "MEDIUM",
                "message": "Driver fatigue detected"
            },
            "LOW": {
                "action": "No action required",
                "severity": "LOW",
                "message": "Driver condition normal"
            }
        }

        return decisions[state]

    # ------------------------
    # CONFIDENCE INTERPRETATION
    # ------------------------
    def _interpret_confidence(self, confidence: float) -> str:

        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.55:
            return "Moderate"
        return "Low"

    # ------------------------
    # LOCAL FEATURE CONTRIBUTION
    # ------------------------
    def _get_top_contributors(self, input_df: pd.DataFrame):

        if self.feature_importances is None:
            return []

        feature_values = input_df.iloc[0].values
        feature_names = input_df.columns
        importances = self.feature_importances

        weighted_scores = []

        for name, value, importance in zip(
            feature_names,
            feature_values,
            importances
        ):
            contribution = abs(value * importance)

            weighted_scores.append({
                "feature": name,
                "feature_value": float(value),
                "global_importance": float(round(importance, 4)),
                "local_contribution_score": float(round(contribution, 4))
            })

        weighted_scores.sort(
            key=lambda x: x["local_contribution_score"],
            reverse=True
        )

        return weighted_scores[:3]

    # ------------------------
    # EXPLANATION ENGINE
    # ------------------------
    def _generate_explanations(
        self,
        data: Dict,
        prediction: str,
        confidence: float
    ) -> List[str]:

        reasons = []

        if data["Fatigue"] > 6:
            reasons.append("Fatigue level is high")

        if data["Alertness"] < 0.5:
            reasons.append("Alertness level is low")

        if data["HR"] > 100:
            reasons.append("Heart rate indicates stress")

        if (data["prev_alertness"] - data["Alertness"]) > 0.25:
            reasons.append("Rapid drop in alertness detected")

        if prediction == "Drowsy":
            reasons.append("ML model detected drowsiness pattern")

        reasons.append(
            f"Prediction confidence: {round(confidence * 100, 2)}%"
        )

        reasons.append(
            f"Confidence category: {self._interpret_confidence(confidence)}"
        )

        return reasons

    # ------------------------
    # METRICS
    # ------------------------
    def get_metrics(self):

        with self._metrics_lock:
            total = self.total_requests
            drowsy = self.total_drowsy
            total_latency = self.total_latency

        avg_latency = (
            total_latency / total
            if total > 0 else 0
        )

        return {
            "total_requests": total,
            "drowsy_predictions": drowsy,
            "average_latency_ms": round(avg_latency * 1000, 2)
        }

    # ------------------------
    # DIAGNOSTICS
    # ------------------------
    def diagnostics(self):

        with self._metrics_lock:
            total = self.total_requests
            drowsy = self.total_drowsy

        return {
            "model_loaded": self.model is not None,
            "total_requests": total,
            "drowsy_ratio": (
                drowsy / total if total > 0 else 0
            ),
            "model_version": self.model_version,
            "model_type": self.model_type
        }
