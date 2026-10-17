#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_validate_isolationforest.py

Amaç
-----
- train_ads.csv ile label-free IsolationForest eğitmek
- test_set_1.csv ve test_set_2.csv üzerinde validation yapmak
- scored csv dosyaları üretmek

Not
---
Bu problem unsupervised olduğu için klasik supervised metricler
(accuracy, F1, ROC AUC) hesaplanmaz.

Bunun yerine şu çıktılar verilir:
- anomaly / conformity skorları
- predicted inlier/outlier oranı
- train dağılımına göre percentile-based quality tiers
- test setlerinin train referans uzayına benzerliği

Gereken paketler
----------------
pip install pandas numpy scikit-learn scipy joblib

Beklenen dosyalar
-----------------
train_ads.csv
test_set_1.csv
test_set_2.csv
"""

import os
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent

TRAIN_FILE = CURRENT_DIR / "train_ads.csv"
TEST1_FILE = CURRENT_DIR / "test_set_1.csv"
TEST2_FILE = CURRENT_DIR / "test_set_2.csv"

MODEL_FILE = CURRENT_DIR / "isolationforest_ads.joblib"
SUMMARY_FILE = CURRENT_DIR / "isolationforest_validation_summary.json"

TRAIN_SCORED_FILE = CURRENT_DIR / "train_ads_scored.csv"
TEST1_SCORED_FILE = CURRENT_DIR / "test_set_1_scored.csv"
TEST2_SCORED_FILE = CURRENT_DIR / "test_set_2_scored.csv"

RANDOM_STATE = 42
CONTAMINATION = 0.10

# ---------------------------------------------------------------------
# FEATURE LIST
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "is_xray",
    "is_cryo_em",
    "is_nmr",
    "resolution_A",
    "deposit_year",
    "release_year",
    "age_years",
    "protein_entity_count",
    "total_polymer_entity_count",
    "assembly_count",
    "deposited_model_count",
    "deposited_atom_count",
    "deposited_polymer_monomer_count",
    "deposited_nonpolymer_entity_count",
    "atoms_per_residue",
    "ligands_per_protein_entity",
    "r_free",
    "r_work",
    "r_observed",
    "clashscore",
    "ramachandran_outliers_pct",
    "rotamer_outliers_pct",
    "rsrz_outliers_pct",
    "dcc_r",
    "dcc_rfree",
]

ID_COLS_CANDIDATES = ["pdb_id", "source_file", "exp_method", "structure_url"]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_feature_columns(df: pd.DataFrame, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik feature kolonları: {missing}")


def get_id_cols(df: pd.DataFrame):
    return [c for c in ID_COLS_CANDIDATES if c in df.columns]


def build_pipeline(contamination=CONTAMINATION, random_state=RANDOM_STATE):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", IsolationForest(
            n_estimators=400,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipe


def fit_pipeline(train_df: pd.DataFrame, feature_cols):
    X_train = train_df[feature_cols].copy()
    pipe = build_pipeline()
    pipe.fit(X_train)
    return pipe


def score_dataset(df: pd.DataFrame, pipe: Pipeline, feature_cols, train_score_quantiles=None):
    """
    decision_function:
        yüksek -> daha normal / referansa daha yakın
        düşük  -> daha anormal / referanstan uzak

    predict:
        1  -> inlier
       -1  -> outlier
    """
    X = df[feature_cols].copy()

    preds = pipe.predict(X)
    decision_scores = pipe.decision_function(X)
    raw_scores = pipe.named_steps["model"].score_samples(
        pipe.named_steps["imputer"].transform(X)
    )

    out = df.copy()
    out["if_pred"] = preds
    out["if_is_inlier"] = (preds == 1).astype(int)
    out["if_is_outlier"] = (preds == -1).astype(int)

    out["if_decision_score"] = decision_scores
    out["if_raw_score"] = raw_scores

    # 0-1 normalize: dataset-independent olmak için train quantile'ları ile normalize etmek daha doğru
    if train_score_quantiles is not None:
        q05 = train_score_quantiles["q05"]
        q95 = train_score_quantiles["q95"]
    else:
        q05 = np.quantile(decision_scores, 0.05)
        q95 = np.quantile(decision_scores, 0.95)

    denom = max(q95 - q05, 1e-8)
    conformity_01 = (decision_scores - q05) / denom
    conformity_01 = np.clip(conformity_01, 0, 1)
    out["conformity_score_0_1"] = conformity_01

    # train quantile tabanlı tier
    if train_score_quantiles is not None:
        q20 = train_score_quantiles["q20"]
        q50 = train_score_quantiles["q50"]
        q80 = train_score_quantiles["q80"]
    else:
        q20 = np.quantile(decision_scores, 0.20)
        q50 = np.quantile(decision_scores, 0.50)
        q80 = np.quantile(decision_scores, 0.80)

    def tier_fn(x):
        if x < q20:
            return "low"
        elif x < q50:
            return "moderate"
        elif x < q80:
            return "acceptable"
        else:
            return "high"

    out["conformity_tier"] = out["if_decision_score"].apply(tier_fn)
    return out


def summarize_scores(df: pd.DataFrame, dataset_name: str):
    summary = {
        "dataset": dataset_name,
        "n": int(len(df)),
        "inlier_count": int((df["if_pred"] == 1).sum()),
        "outlier_count": int((df["if_pred"] == -1).sum()),
        "inlier_ratio": float((df["if_pred"] == 1).mean()),
        "outlier_ratio": float((df["if_pred"] == -1).mean()),
        "decision_score_mean": float(df["if_decision_score"].mean()),
        "decision_score_std": float(df["if_decision_score"].std()),
        "decision_score_min": float(df["if_decision_score"].min()),
        "decision_score_q25": float(df["if_decision_score"].quantile(0.25)),
        "decision_score_median": float(df["if_decision_score"].median()),
        "decision_score_q75": float(df["if_decision_score"].quantile(0.75)),
        "decision_score_max": float(df["if_decision_score"].max()),
        "conformity_score_mean_0_1": float(df["conformity_score_0_1"].mean()),
        "conformity_score_std_0_1": float(df["conformity_score_0_1"].std()),
        "tier_counts": df["conformity_tier"].value_counts(dropna=False).to_dict(),
    }
    return summary


def compare_to_train(train_df: pd.DataFrame, test_df: pd.DataFrame, test_name: str):
    """
    Unsupervised validation comparison:
    - KS testi: score dağılımları benzer mi?
    - mean shift
    """
    ks_stat, ks_p = ks_2samp(train_df["if_decision_score"], test_df["if_decision_score"])
    result = {
        "dataset": test_name,
        "vs_train_decision_score_mean_diff": float(test_df["if_decision_score"].mean() - train_df["if_decision_score"].mean()),
        "vs_train_conformity_score_mean_diff": float(test_df["conformity_score_0_1"].mean() - train_df["conformity_score_0_1"].mean()),
        "ks_statistic_decision_score": float(ks_stat),
        "ks_pvalue_decision_score": float(ks_p),
    }
    return result


def save_scored(df: pd.DataFrame, out_file: Path):
    df.to_csv(out_file, index=False)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("[INFO] Loading datasets...")
    train_df = load_csv(TRAIN_FILE)
    test1_df = load_csv(TEST1_FILE)
    test2_df = load_csv(TEST2_FILE)

    ensure_feature_columns(train_df, FEATURE_COLS)
    ensure_feature_columns(test1_df, FEATURE_COLS)
    ensure_feature_columns(test2_df, FEATURE_COLS)

    print(f"[INFO] train shape   : {train_df.shape}")
    print(f"[INFO] test1 shape   : {test1_df.shape}")
    print(f"[INFO] test2 shape   : {test2_df.shape}")

    print("[INFO] Fitting pipeline...")
    pipe = fit_pipeline(train_df, FEATURE_COLS)

    # Train score quantile'larını alıp testte de aynı standardı kullan
    train_tmp = train_df.copy()
    train_tmp["if_decision_score"] = pipe.decision_function(train_df[FEATURE_COLS])

    train_quantiles = {
        "q05": float(train_tmp["if_decision_score"].quantile(0.05)),
        "q20": float(train_tmp["if_decision_score"].quantile(0.20)),
        "q50": float(train_tmp["if_decision_score"].quantile(0.50)),
        "q80": float(train_tmp["if_decision_score"].quantile(0.80)),
        "q95": float(train_tmp["if_decision_score"].quantile(0.95)),
    }

    print("[INFO] Scoring datasets...")
    train_scored = score_dataset(train_df, pipe, FEATURE_COLS, train_quantiles)
    test1_scored = score_dataset(test1_df, pipe, FEATURE_COLS, train_quantiles)
    test2_scored = score_dataset(test2_df, pipe, FEATURE_COLS, train_quantiles)

    print("[INFO] Saving scored CSV files...")
    save_scored(train_scored, TRAIN_SCORED_FILE)
    save_scored(test1_scored, TEST1_SCORED_FILE)
    save_scored(test2_scored, TEST2_SCORED_FILE)

    print("[INFO] Saving model...")
    joblib.dump({
        "pipeline": pipe,
        "feature_cols": FEATURE_COLS,
        "train_quantiles": train_quantiles,
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
    }, MODEL_FILE)

    print("[INFO] Creating summaries...")
    train_summary = summarize_scores(train_scored, "train_ads")
    test1_summary = summarize_scores(test1_scored, "test_set_1")
    test2_summary = summarize_scores(test2_scored, "test_set_2")

    test1_vs_train = compare_to_train(train_scored, test1_scored, "test_set_1")
    test2_vs_train = compare_to_train(train_scored, test2_scored, "test_set_2")

    all_summary = {
        "model": {
            "type": "IsolationForest",
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
            "n_features": len(FEATURE_COLS),
            "feature_cols": FEATURE_COLS,
        },
        "train_quantiles": train_quantiles,
        "train_summary": train_summary,
        "test1_summary": test1_summary,
        "test2_summary": test2_summary,
        "comparisons_vs_train": {
            "test_set_1": test1_vs_train,
            "test_set_2": test2_vs_train,
        }
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("UNSUPERVISED VALIDATION SUMMARY")
    print("=" * 80)

    for s in [train_summary, test1_summary, test2_summary]:
        print(f"\nDataset: {s['dataset']}")
        print(f"  n                     : {s['n']}")
        print(f"  inlier_ratio          : {s['inlier_ratio']:.4f}")
        print(f"  outlier_ratio         : {s['outlier_ratio']:.4f}")
        print(f"  decision_score_mean   : {s['decision_score_mean']:.4f}")
        print(f"  decision_score_std    : {s['decision_score_std']:.4f}")
        print(f"  conformity_mean_0_1   : {s['conformity_score_mean_0_1']:.4f}")
        print(f"  conformity_std_0_1    : {s['conformity_score_std_0_1']:.4f}")
        print(f"  tier_counts           : {s['tier_counts']}")

    print("\nComparisons vs train:")
    for name, comp in all_summary["comparisons_vs_train"].items():
        print(f"\n{name}")
        print(f"  mean_diff_decision_score    : {comp['vs_train_decision_score_mean_diff']:.4f}")
        print(f"  mean_diff_conformity_0_1    : {comp['vs_train_conformity_score_mean_diff']:.4f}")
        print(f"  KS statistic                : {comp['ks_statistic_decision_score']:.4f}")
        print(f"  KS p-value                  : {comp['ks_pvalue_decision_score']:.6f}")

    print("\nFiles saved:")
    print(f"  Model              : {MODEL_FILE}")
    print(f"  Summary            : {SUMMARY_FILE}")
    print(f"  Train scored       : {TRAIN_SCORED_FILE}")
    print(f"  Test1 scored       : {TEST1_SCORED_FILE}")
    print(f"  Test2 scored       : {TEST2_SCORED_FILE}")

    # En yüksek ve en düşük skor örnekleri
    id_cols_train = get_id_cols(train_scored)
    show_cols = id_cols_train + ["if_decision_score", "conformity_score_0_1", "conformity_tier"]

    print("\nTop 10 train highest conformity:")
    print(train_scored[show_cols].sort_values("if_decision_score", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 train lowest conformity:")
    print(train_scored[show_cols].sort_values("if_decision_score", ascending=True).head(10).to_string(index=False))


if __name__ == "__main__":
    main()