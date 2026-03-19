"""
Random Forest classifier for LLM authorship attribution.

Based on the stylometric approach from Opara (2024) and Kumarage et al. (2023).
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.analysis.constants import CORE_FEATURES


class AuthorshipClassifier:
    """RF-based classifier for model attribution."""

    def __init__(self, classifier_params: Optional[Dict[str, Any]] = None):
        self.classifier_params = classifier_params or {}
        self.classifier = RandomForestClassifier(**self.classifier_params)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        self._feature_names: Optional[List[str]] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, List[str]]) -> "AuthorshipClassifier":
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values

        y_enc = self.label_encoder.fit_transform(y)
        X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y_enc)
        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.scaler.transform(X)
        return self.label_encoder.inverse_transform(self.classifier.predict(X))

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.classifier.predict_proba(self.scaler.transform(X))

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, List[str]]) -> Dict[str, Any]:
        """Run prediction and return accuracy, F1, confusion matrix, etc."""
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average="macro"),
            "recall_macro": recall_score(y, y_pred, average="macro"),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred),
            "classes": self.label_encoder.classes_.tolist(),
        }

    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[str]],
        cv: int = 5,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[str, float]:
        """K-fold CV with optional GroupKFold (e.g. by prompt_id to avoid leakage).

        Uses a pipeline so the scaler is fit per fold.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_enc = self.label_encoder.fit_transform(y)

        pipe = make_pipeline(StandardScaler(), self.classifier)

        if groups is not None:
            splitter = GroupKFold(n_splits=cv)
            scores = cross_val_score(pipe, X, y_enc, cv=splitter,
                                     groups=groups, scoring="f1_macro")
        else:
            scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="f1_macro")

        return {
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
            "all_scores": scores.tolist(),
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self._is_fitted:
            return None

        return (pd.DataFrame({"feature": self._feature_names,
                             "importance": self.classifier.feature_importances_})
                .sort_values("importance", ascending=False))

    def run_cross_genre_transfer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Leave-one-genre-out transfer test.

        Trains on all genres except one, tests on the held-out genre.
        Checks whether model signatures persist across genre boundaries.
        """
        genres = sorted(df["genre"].unique())

        feat_cols = [c for c in CORE_FEATURES if c in df.columns]

        results = []
        for held_out in genres:
            train = df[df["genre"] != held_out]
            test = df[df["genre"] == held_out]

            clf = AuthorshipClassifier(classifier_params={'n_estimators':1000, 'random_state': 42})
            clf.fit(train[feat_cols], train["label"])
            metrics = clf.evaluate(test[feat_cols], test["label"])

            results.append({
                "held_out_genre": held_out,
                "train_genres": ", ".join(sorted(train["genre"].unique())),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "n_train": len(train),
                "n_test": len(test),
            })

        return pd.DataFrame(results).sort_values("held_out_genre").reset_index(drop=True)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "classifier": self.classifier,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> "AuthorshipClassifier":
        import joblib
        data = joblib.load(path)
        inst = cls()
        inst.classifier = data["classifier"]
        inst.scaler = data["scaler"]
        inst.label_encoder = data["label_encoder"]
        inst._feature_names = data["feature_names"]
        inst._is_fitted = data["is_fitted"]
        return inst
