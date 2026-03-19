"""Plots for the stylometric analysis pipeline."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

DPI = 150


class StylometricPlotter:

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("seaborn-whitegrid")

    def _save(self, fig: plt.Figure, path: Optional[str]) -> None:
        plt.tight_layout()
        if path:
            fig.savefig(path, dpi=DPI, bbox_inches="tight")

    @staticmethod
    def _feature_cols(df: pd.DataFrame, label: str) -> List[str]:
        return [c for c in df.columns if c != label]

    def _scatter_by_label(self, ax, points, labels, label_col):
        for lab in labels[label_col].unique():
            mask = labels[label_col] == lab
            ax.scatter(points[mask, 0], points[mask, 1],
                       label=lab, alpha=0.7, s=60)
        ax.legend()

    def plot_feature_comparison(
        self,
        df: pd.DataFrame,
        features: List[str],
        group_by: str = "label",
        plot_type: str = "box",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Side-by-side box/violin plots for a set of features."""
        n = len(features)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        plot_fn = sns.violinplot if plot_type == "violin" else sns.boxplot
        for ax, feat in zip(axes, features):
            plot_fn(data=df, x=group_by, y=feat, ax=ax)
            ax.set_title(feat)
            ax.tick_params(axis='x', rotation=45)

        self._save(fig, save_path)
        return fig

    def plot_pca(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        n_components: int = 2,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, PCA]:
        """PCA scatter — features are z-scored first so high-magnitude
        columns like paragraph_count don't dominate."""
        cols = self._feature_cols(df, label_col)
        X = StandardScaler().fit_transform(df[cols].values)

        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=self.figsize)
        self._scatter_by_label(ax, proj, df, label_col)
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
        ax.set_title("PCA: Stylometric Feature Space")

        self._save(fig, save_path)
        return fig, pca

    def plot_tsne(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        perplexity: int = 30,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """t-SNE scatter. PCA init + auto LR for stability."""
        cols = self._feature_cols(df, label_col)
        X = StandardScaler().fit_transform(df[cols].values)

        proj = TSNE(
            n_components=2, perplexity=perplexity,
            init="pca", learning_rate="auto", random_state=42,
        ).fit_transform(X)

        fig, ax = plt.subplots(figsize=self.figsize)
        self._scatter_by_label(ax, proj, df, label_col)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE: Stylometric Feature Space")

        self._save(fig, save_path)
        return fig

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Model Identification Confusion Matrix")

        self._save(fig, save_path)
        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 19,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.barplot(data=importance_df.head(top_n),
                    y="feature", x="importance", ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.set_xlabel("Importance")

        self._save(fig, save_path)
        return fig
