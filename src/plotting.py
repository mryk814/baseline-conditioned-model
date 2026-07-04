from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_predicted_vs_true(results: pd.DataFrame, output_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(results["true_delta_y"], results["pred_delta_y"], alpha=0.65)
    low = min(results["true_delta_y"].min(), results["pred_delta_y"].min())
    high = max(results["true_delta_y"].max(), results["pred_delta_y"].max())
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("True delta_y")
    ax.set_ylabel("Predicted delta_y")
    ax.set_title("Predicted vs true material-property change")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig


def plot_model_comparison(metrics: dict[str, dict[str, float]], output_path: str | Path | None = None):
    frame = pd.DataFrame(metrics).T.sort_values("rmse_delta_y")
    fig, ax = plt.subplots(figsize=(7, 4))
    frame["rmse_delta_y"].plot(kind="bar", ax=ax, color="#4f7cac")
    ax.set_ylabel("RMSE delta_y")
    ax.set_title("Model comparison")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig


def plot_local_contributions(contributions: dict[str, float], output_path: str | Path | None = None):
    frame = pd.Series(contributions).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#b14d4d" if value < 0 else "#3f8f68" for value in frame]
    frame.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Contribution to delta_y")
    ax.set_title("Local linear contributions")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig


def plot_uncertainty_vs_error(results: pd.DataFrame, output_path: str | Path | None = None):
    frame = results.copy()
    frame["absolute_error"] = (frame["true_delta_y"] - frame["pred_delta_y"]).abs()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(frame["std_delta_y"], frame["absolute_error"], alpha=0.7, color="#7d6b91")
    ax.set_xlabel("Estimated std_delta_y")
    ax.set_ylabel("Absolute error")
    ax.set_title("Uncertainty vs absolute error")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig


def plot_candidate_ranking(scored_candidates: pd.DataFrame, output_path: str | Path | None = None):
    frame = scored_candidates.sort_values("risk_adjusted_score", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(frame["candidate_name"], frame["risk_adjusted_score"], color="#3f8f68")
    ax.scatter(frame["true_delta_y"], frame["candidate_name"], color="#202020", label="true delta_y", zorder=3)
    ax.set_xlabel("Risk-adjusted score")
    ax.set_title("Candidate ranking comparison")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig
