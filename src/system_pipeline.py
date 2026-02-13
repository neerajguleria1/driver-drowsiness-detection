import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DriverSafetySystem:
    """
    Production-grade Decision Intelligence System.
    ML is only ONE component.
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

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # ------------------------
    # HEALTH CHECK
    # ------------------------
    def health_check(self):

        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            dummy_data = {feature: 0 for feature in self.MODEL_FEATURES}

            # override realistic values
            dummy_data.update({
                "Alertness": 1,
                "Seatbelt": 1,
                "HR": 70,
                "prev_alertness": 1
            })

            dummy = pd.DataFrame([dummy_data])[self.MODEL_FEATURES]

            self.model.predict(dummy)

        except Exception as e:
            raise RuntimeError(f"Model unhealthy: {e}")

    # ------------------------
    # MAIN ENTRY
    # ------------------------
    def analyze(self, input_data: Dict) -> Dict:

        self._validate_input(input_data)

        features = self._prepare_features(input_data)

        try:
            probabilities = self.model.predict_proba(features)[0]
        except Exception as e:
            raise RuntimeError(f"Inference failure: {e}")

        pred_index = int(np.argmax(probabilities))

        ml_prediction = "Drowsy" if pred_index == 1 else "Alert"
        ml_confidence = float(probabilities[pred_index])
        ml_confidence = min(max(ml_confidence, 0.0), 1.0)

        risk_score = self._compute_risk_score(input_data, ml_confidence)
        risk_state = self._map_risk_state(risk_score)
        decision = self._decision_engine(risk_state)
        explanations = self._generate_explanations(
            input_data,
            ml_prediction,
            ml_confidence
        )

        return {
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "risk_score": risk_score,
            "risk_state": risk_state,
            "decision": decision,
            "explanations": explanations
        }

    # ------------------------
    # VALIDATION
    # ------------------------
    def _validate_input(self, data: Dict):

        required = set(self.MODEL_FEATURES)
        provided = set(data.keys())

        if required != provided:
            raise ValueError("Input schema mismatch detected")

        if not 0 <= data["Alertness"] <= 1:
            raise ValueError("Alertness must be between 0 and 1")

        if not 0 <= data["Speed"] <= 200:
            raise ValueError("Speed must be between 0 and 200")

        if not 30 <= data["HR"] <= 200:
            raise ValueError("Heart rate out of range")

        if not 0 <= data["Fatigue"] <= 10:
            raise ValueError("Fatigue must be between 0 and 10")

    # ------------------------
    # FEATURE PREP
    # ------------------------
    def _prepare_features(self, data: Dict) -> pd.DataFrame:
        return pd.DataFrame([data])[self.MODEL_FEATURES]

    # ------------------------
    # RISK ENGINE
    # ------------------------
    def _compute_risk_score(self, data: Dict, confidence: float) -> int:

        score = 0

        if data["Fatigue"] > 6:
            score += 30

        if data["Alertness"] < 0.5:
            score += 25

        if data["HR"] > 100:
            score += 15

        if confidence < 0.55:
            score += 25

        alert_drop = data["prev_alertness"] - data["Alertness"]

        if alert_drop > 0.25:
            score += 20

        if abs(alert_drop) > 0.5:
            score += 15

        if np.isnan(confidence):
            score += 30

        if data["Speed"] == 0 and data["Fatigue"] > 7:
            score += 20

        # Signal smoothing
        self.previous_score = getattr(self, "previous_score", score)
        score = int(0.7 * self.previous_score + 0.3 * score)
        self.previous_score = score

        return max(0, min(score, 100))

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
    # EXPLAINABILITY
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

        if confidence < 0.55:
            reasons.append(
                "Model uncertainty detected — driver may be transitioning to drowsy"
            )

        if (data["prev_alertness"] - data["Alertness"]) > 0.25:
            reasons.append("Rapid drop in alertness detected")

        if prediction == "Drowsy":
            reasons.append("ML model detected drowsiness pattern")

        return reasons
