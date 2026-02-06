import joblib
import numpy as np
import pandas as pd
from typing import Dict, List


class DriverSafetySystem:
    """
    Production-style Decision Intelligence System.
    ML is only ONE component.
    """

    # ✅ Feature contract (VERY IMPORTANT)
    EXPECTED_FEATURES = [
        "Speed",
        "Alertness",
        "Seatbelt",
        "HR",   # model expects HR
        "Fatigue",
        "speed_change",
        "prev_alertness"
    ]

    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
            print("Model loaded successfully")

            # Optional — verify schema against model
            if hasattr(self.model, "feature_names_in_"):
                print("Model expects:", self.model.feature_names_in_)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # ------------------------
    # HEALTH CHECK
    # ------------------------
    def health_check(self):
        if self.model is None:
            raise RuntimeError("Model not loaded")

    # ------------------------
    # MAIN ENTRY
    # ------------------------
    def analyze(self, input_data: Dict) -> Dict:

        self._validate_input(input_data)

        features = self._prepare_features(input_data)

        probabilities = self.model.predict_proba(features)[0]
        pred_index = int(np.argmax(probabilities))

        ml_prediction = "Drowsy" if pred_index == 1 else "Alert"
        ml_confidence = float(probabilities[pred_index])

        ml_confidence = min(max(ml_confidence, 0.0), 1.0)

        risk_score = self._compute_risk_score(input_data, ml_confidence)
        risk_state = self._map_risk_state(risk_score)
        decision = self._decision_engine(risk_state)
        explanations = self._generate_explanations(input_data, ml_prediction,ml_confidence)

        return {
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "risk_score": risk_score,
            "risk_state": risk_state,
            "decision": decision,
            "explanations": explanations
        }

    # ------------------------
    # VALIDATION (ENGINEER LEVEL)
    # ------------------------
    def _validate_input(self, data: Dict):

        if not 0 <= data["Alertness"] <= 1:
            raise ValueError("Alertness must be between 0 and 1")

        if data["Speed"] < 0 or data["Speed"] > 200:
            raise ValueError("Speed must be between 0 and 200")

        if data["HR"] if "HR" in data else data["Heart_rate"] < 30:
            raise ValueError("Heart rate is unrealistically low")

        if data["Fatigue"] < 0 or data["Fatigue"] > 10:
            raise ValueError("Fatigue must be between 0 and 10")

    # ------------------------
    # PREPARE FEATURES
    # ------------------------
    def _prepare_features(self, data: Dict) -> pd.DataFrame:

        df = pd.DataFrame([{
            "Speed": data["Speed"],
            "Alertness": data["Alertness"],
            "Seatbelt": data["Seatbelt"],
            "HR": data["Heart_rate"],   # schema translation
            "Fatigue": data["Fatigue"],
            "speed_change": data["speed_change"],
            "prev_alertness": data["prev_alertness"]
        }])

        # enforce column order
        df = df[self.EXPECTED_FEATURES]

        return df

    # ------------------------
    # RISK ENGINE
    # ------------------------
    def _compute_risk_score(self, data: Dict, confidence: float) -> int:

        score = 0

        if data["Fatigue"] > 6:
            score += 30

        if data["Alertness"] < 0.5:
            score += 25

        if data["Heart_rate"] > 100:
            score += 15

        if confidence < 0.55:
            score += 25

        alert_drop = data["prev_alertness"] - data["Alertness"]

        if alert_drop > 0.25:
            score += 20

        #  sensor anomaly detection
        if abs(alert_drop) > 0.5:
            score += 15
        
        return int(min(max(score, 0), 100))

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
    def _generate_explanations(self, data: Dict, prediction: str,confidence:float) -> List[str]:

        reasons = []

        if data["Fatigue"] > 6:
            reasons.append("Fatigue level is high")

        if data["Alertness"] < 0.5:
            reasons.append("Alertness level is low")

        if data["Heart_rate"] > 100:
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