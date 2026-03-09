import joblib
import numpy as np
import pandas as pd
import logging
import threading
import time
from typing import Dict, List, Tuple
import os
import json
import queue
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
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

            # ------------------------
            # MODEL LOCK (FIX YOUR BUG)
            # ------------------------
            self.model_lock = threading.Lock()

            # ------------------------
            # AUDIT LOGGING SETUP
            # ------------------------
            os.makedirs("logs", exist_ok=True)

            log_queue = queue.Queue(-1)

            queue_handler = QueueHandler(log_queue)

            file_handler = RotatingFileHandler(
                "logs/driver_audit.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter('%(message)s'))

            self.listener = QueueListener(log_queue, file_handler)
            self.listener.start()

            self.audit_logger = logging.getLogger("audit")
            self.audit_logger.setLevel(logging.INFO)
            self.audit_logger.addHandler(queue_handler)

            # Shadow model for A/B testing
            self.shadow_model = None
            self.shadow_metadata = None

            # Drift detection buffers
            self.live_feature_buffer = []
            self.prediction_buffer = []
            self.baseline_stats = {}
            self.baseline_drowsy_ratio = 0.0
            self._drift_lock = threading.Lock()

            # Performance monitoring
            self.prediction_counts = {"Alert": 0, "Drowsy": 0}
            self.confidence_scores = []

            # Circuit breaker for reliability
            self.failure_count = 0
            self.circuit_open = False
            self.last_failure_time = 0
            self.circuit_threshold = 5
            self.circuit_timeout = 60

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # ------------------------
    # PERFORMANCE MONITORING
    # ------------------------
    def performance_report(self) -> Dict:
        """Generate performance metrics report"""
        with self._metrics_lock:
            avg_conf = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
            total_preds = sum(self.prediction_counts.values())

        return {
            "prediction_distribution": self.prediction_counts,
            "avg_confidence": round(avg_conf, 3),
            "total_predictions": total_preds,
            "confidence_scores_tracked": len(self.confidence_scores)
        }

    # ------------------------
    # MODEL VERSIONING
    # ------------------------
    def load_model_version(self, version: str):
        """Dynamically load and switch model without restart"""
        model_path = f"models/{version}/model.pkl"
        metadata_path = f"models/{version}/metadata.json"

        if not os.path.exists(model_path):
            raise ValueError(f"Model version {version} not found")

        new_model = joblib.load(model_path)

        with open(metadata_path) as f:
            new_metadata = json.load(f)

        # Validate feature compatibility
        if set(new_model.feature_names_in_) != set(self.MODEL_FEATURES):
            raise ValueError("Feature mismatch between models")

        # Atomic switch
        with self.model_lock:
            self.model = new_model
            self.model_version = version
            logger.info(f"Model switched to {version}")

    def load_shadow_model(self, version: str):
        """Load model for shadow testing without switching"""
        model_path = f"models/{version}/model.pkl"
        metadata_path = f"models/{version}/metadata.json"

        if not os.path.exists(model_path):
            raise ValueError(f"Shadow model version {version} not found")

        self.shadow_model = joblib.load(model_path)

        with open(metadata_path) as f:
            self.shadow_metadata = json.load(f)

        logger.info(f"Shadow model loaded: {version}")

    def _run_shadow_prediction(self, features: pd.DataFrame) -> Dict:
        """Run prediction on shadow model and log comparison"""
        if self.shadow_model is None:
            return None

        try:
            shadow_prob = self.shadow_model.predict_proba(features)[0]
            shadow_pred = "Drowsy" if np.argmax(shadow_prob) == 1 else "Alert"
            shadow_conf = float(shadow_prob[np.argmax(shadow_prob)])

            return {
                "prediction": shadow_pred,
                "confidence": shadow_conf
            }
        except Exception as e:
            logger.warning(f"Shadow prediction failed: {e}")
            return None

    # ------------------------
    # DRIFT DETECTION
    # ------------------------
    def set_baseline_stats(self, baseline_df: pd.DataFrame, baseline_predictions: List[str]):
        """Initialize baseline statistics for drift detection"""
        for feature in self.MODEL_FEATURES:
            self.baseline_stats[feature] = {
                "mean": baseline_df[feature].mean(),
                "std": baseline_df[feature].std()
            }
        self.baseline_drowsy_ratio = sum(1 for p in baseline_predictions if p == "Drowsy") / len(baseline_predictions)
        logger.info("Baseline statistics initialized")

    def detect_drift(self) -> Dict:
        """Detect data and prediction drift"""
        with self._drift_lock:
            if len(self.live_feature_buffer) < 200:
                return {"status": "Insufficient data"}

            live_df = pd.DataFrame(self.live_feature_buffer)
            drift_report = {}

            # Feature drift detection
            for feature in self.MODEL_FEATURES:
                if feature not in self.baseline_stats:
                    continue

                live_mean = live_df[feature].mean()
                live_std = live_df[feature].std()
                baseline_mean = self.baseline_stats[feature]["mean"]
                baseline_std = self.baseline_stats[feature]["std"]

                mean_shift = abs(live_mean - baseline_mean) / (abs(baseline_mean) + 1e-6)
                std_shift = abs(live_std - baseline_std) / (abs(baseline_std) + 1e-6)

                if mean_shift > 0.2:
                    drift_report[f"{feature}_mean"] = f"Mean shift: {mean_shift:.2%}"

                if std_shift > 0.2:
                    drift_report[f"{feature}_std"] = f"Variance shift: {std_shift:.2%}"

            # Prediction distribution drift
            live_drowsy_ratio = sum(1 for p in self.prediction_buffer if p == "Drowsy") / len(self.prediction_buffer)
            pred_drift = abs(live_drowsy_ratio - self.baseline_drowsy_ratio)

            if pred_drift > 0.15:
                drift_report["prediction_distribution"] = f"Drowsy ratio shift: {self.baseline_drowsy_ratio:.2%} → {live_drowsy_ratio:.2%}"

            if drift_report:
                self.audit_logger.warning(json.dumps({
                    "event": "drift_detected",
                    "timestamp": time.time(),
                    "details": drift_report
                }))

            return {
                "status": "Drift Warning" if drift_report else "Stable",
                "details": drift_report
            }

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
    def analyze(self, input_data: Dict, trace_id: str, max_retries: int = 3) -> Dict:

        start_time = time.time()

        # Circuit breaker check
        if self.circuit_open:
            if time.time() - self.last_failure_time > self.circuit_timeout:
                self.circuit_open = False
                self.failure_count = 0
                logger.info("Circuit breaker reset")
            else:
                logger.warning(f"[{trace_id}] Circuit breaker open, using fallback")
                return self._safe_fallback_response(input_data, trace_id, "Circuit breaker open")

        self._validate_input(input_data)
        features = self._prepare_features(input_data)

        # Retry logic for model inference
        for attempt in range(max_retries):
            try:
                with self.model_lock:
                    probabilities = self.model.predict_proba(features)[0]
                self.failure_count = 0
                break
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.circuit_threshold:
                    self.circuit_open = True
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                
                if attempt == max_retries - 1:
                    logger.error(f"Model inference failed after {max_retries} attempts: {e}")
                    return self._safe_fallback_response(input_data, trace_id, str(e))
                time.sleep(0.1 * (attempt + 1))

        pred_index = int(np.argmax(probabilities))

        ml_prediction = "Drowsy" if pred_index == 1 else "Alert"
        ml_confidence = float(probabilities[pred_index]) ** 0.7
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

        # Shadow model comparison
        shadow_result = self._run_shadow_prediction(features)
        if shadow_result:
            self.audit_logger.info(json.dumps({
                "trace_id": trace_id,
                "shadow_comparison": {
                    "main_prediction": ml_prediction,
                    "shadow_prediction": shadow_result["prediction"],
                    "match": ml_prediction == shadow_result["prediction"]
                }
            }))

        # Thread-safe metrics update
        with self._metrics_lock:
            self.total_requests += 1
            self.total_latency += latency
            if ml_prediction == "Drowsy":
                self.total_drowsy += 1

        # Track for drift detection
        with self._drift_lock:
            self.live_feature_buffer.append(input_data)
            self.prediction_buffer.append(ml_prediction)
            if len(self.live_feature_buffer) > 1000:
                self.live_feature_buffer.pop(0)
                self.prediction_buffer.pop(0)

        # Track performance metrics
        with self._metrics_lock:
            self.prediction_counts[ml_prediction] += 1
            self.confidence_scores.append(ml_confidence)
            if len(self.confidence_scores) > 1000:
                self.confidence_scores.pop(0)

        audit_record = {
            "trace_id": trace_id,
            "timestamp": time.time(),
            "input": input_data,
            "ml_prediction": ml_prediction,
            "confidence": ml_confidence,
            "risk_score": risk_score,
            "risk_state": risk_state,
            "decision": decision
        }

        self.audit_logger.info(json.dumps(audit_record))

        if risk_state == "CRITICAL":
            self.audit_logger.info(json.dumps({
                "incident": True,
                "trace_id": trace_id,
                "timestamp": time.time()
            }))

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

    def analyze_batch(self, inputs: List[Dict]) -> List[Dict]:
        if not inputs:
            return []

        start_time = time.time()
        validated = [self._validate_input(data) or data for data in inputs]
        df = pd.DataFrame(validated)[self.MODEL_FEATURES]

        try:
            with self.model_lock:
                probabilities = self.model.predict_proba(df)
        except Exception:
            raise RuntimeError("Batch model inference failure")

        results = [
            self._process_single_result(data, probabilities[i], df.iloc[[i]])
            for i, data in enumerate(validated)
        ]

        self._update_batch_metrics(results, time.time() - start_time)
        return results

    def _process_single_result(self, data: Dict, prob: np.ndarray, single_df: pd.DataFrame) -> Dict:
        pred_index = int(np.argmax(prob))
        ml_prediction = "Drowsy" if pred_index == 1 else "Alert"
        ml_confidence = float(prob[pred_index]) ** 0.7
        ml_confidence = float(np.clip(ml_confidence, 0.0, 1.0))

        risk_score, risk_factors = self._compute_risk_score(data, ml_confidence)
        risk_state = self._map_risk_state(risk_score)

        return {
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "confidence_level": self._interpret_confidence(ml_confidence),
            "risk_score": risk_score,
            "risk_state": risk_state,
            "risk_factors": risk_factors,
            "decision": self._decision_engine(risk_state),
            "top_contributing_features": self._get_top_contributors(single_df),
            "explanations": self._generate_explanations(data, ml_prediction, ml_confidence),
            "model_version": self.model_version,
            "model_type": self.model_type,
        }

    def _update_batch_metrics(self, results: List[Dict], latency: float):
        with self._metrics_lock:
            self.total_requests += len(results)
            self.total_latency += latency
            self.total_drowsy += sum(1 for r in results if r["ml_prediction"] == "Drowsy")
    # ------------------------
    # VALIDATION
    # ------------------------
    def _safe_fallback_response(self, input_data: Dict, trace_id: str, error: str) -> Dict:
        """Return safe fallback response when model fails"""
        logger.warning(f"[{trace_id}] Using fallback response due to: {error}")
        
        # Rule-based fallback logic
        fallback_prediction = "Alert"
        fallback_confidence = 0.5
        
        if input_data["Fatigue"] > 7 or input_data["Alertness"] < 0.4:
            fallback_prediction = "Drowsy"
            fallback_confidence = 0.6
        
        risk_score, risk_factors = self._compute_risk_score(input_data, fallback_confidence)
        risk_state = self._map_risk_state(risk_score)
        
        return {
            "ml_prediction": fallback_prediction,
            "ml_confidence": fallback_confidence,
            "confidence_level": "Low",
            "risk_score": risk_score,
            "risk_state": risk_state,
            "risk_factors": risk_factors,
            "decision": self._decision_engine(risk_state),
            "top_contributing_features": [],
            "explanations": ["Fallback mode: Model unavailable", "Using rule-based prediction"],
            "model_version": "fallback",
            "model_type": "rule_based",
            "inference_latency_ms": 0.0,
            "fallback_mode": True,
            "error": error
        }

    # ------------------------
    # VALIDATION
    # ------------------------
    def _validate_input(self, data: Dict):

        required = set(self.MODEL_FEATURES)
        provided = set(data.keys())

        if not required.issubset(provided):
            raise ValueError("Input schema mismatch detected")

        if data["Seatbelt"] not in [0, 1]:
            raise ValueError("Seatbelt must be 0 or 1")

        if not isinstance(data["Fatigue"], int):
            raise ValueError("Fatigue must be integer")

        if data["Speed"] < 0:
            raise ValueError("Speed cannot be negative")

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
