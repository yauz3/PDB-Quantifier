#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_random_test_set_2.py

Amaç
-----
RCSB PDB'den rastgele 100 PDB ID seçmek, aynı feature'ları üretmek
ve test_set_2.csv olarak kaydetmek.

Özellikler
----------
- Filtre yok
- Rastgele 100 entry seçer
- RCSB core entry API'den feature çeker
- experimental protein olmayanlar veya parse edilemeyenler atlanır
- 100 başarılı satıra ulaşana kadar devam eder

Çıktı
-----
test_set_2.csv
test_set_2_failed.csv

Gereken paketler
----------------
pip install requests pandas numpy
"""

import os
import time
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HEADERS = {"User-Agent": "random-testset-builder/1.0"}
CURRENT_IDS_URL = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
ENTRY_CORE_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

N_TARGET = 100
RANDOM_SEED = 42
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.05

CURRENT_DIR = Path(__file__).resolve().parent
OUT_CSV = CURRENT_DIR / "test_set_2.csv"
FAILED_CSV = CURRENT_DIR / "test_set_2_failed.csv"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_get_json(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def get_nested(d: Any, path: str, default=None):
    cur = d
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                idx = int(part)
                if idx < len(cur):
                    cur = cur[idx]
                else:
                    return default
            except Exception:
                return default
        elif isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
            else:
                return default
        else:
            return default
    return cur


def first_number(x, default=np.nan):
    if x is None:
        return default
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, list) and len(x) > 0:
        for item in x:
            if isinstance(item, (int, float, np.integer, np.floating)):
                return float(item)
    try:
        return float(x)
    except Exception:
        return default


def year_from_date(s: Optional[str]) -> float:
    if not s or not isinstance(s, str):
        return np.nan
    try:
        return float(int(s[:4]))
    except Exception:
        return np.nan


def fetch_current_entry_ids() -> List[str]:
    data = safe_get_json(CURRENT_IDS_URL)
    if not data or not isinstance(data, list):
        raise RuntimeError("Current PDB entry IDs alınamadı.")
    return data


def is_experimental_protein(entry_json: dict) -> bool:
    methodology = str(get_nested(entry_json, "rcsb_entry_info.structure_determination_methodology", "")).lower()
    protein_count = first_number(get_nested(entry_json, "rcsb_entry_info.polymer_entity_count_protein"), default=0)
    return ("experimental" in methodology) and (protein_count > 0)


def extract_entry_features(entry_json: dict) -> Optional[Dict[str, Any]]:
    if not entry_json:
        return None

    if not is_experimental_protein(entry_json):
        return None

    pdb_id = get_nested(entry_json, "rcsb_id", None)
    if pdb_id is None:
        pdb_id = get_nested(entry_json, "entry.id", None)
    if pdb_id is None:
        return None

    exp_methods = get_nested(entry_json, "exptl", [])
    if isinstance(exp_methods, list) and len(exp_methods) > 0:
        exp_method = "; ".join(sorted(set(str(x.get("method", "")) for x in exp_methods if isinstance(x, dict))))
    else:
        exp_method = str(get_nested(entry_json, "rcsb_entry_info.experimental_method", ""))

    resolution = first_number(get_nested(entry_json, "rcsb_entry_info.resolution_combined"))
    deposit_year = year_from_date(get_nested(entry_json, "rcsb_accession_info.deposit_date"))
    release_year = year_from_date(get_nested(entry_json, "rcsb_accession_info.initial_release_date"))

    protein_entity_count = first_number(get_nested(entry_json, "rcsb_entry_info.polymer_entity_count_protein"))
    total_polymer_entity_count = first_number(get_nested(entry_json, "rcsb_entry_info.polymer_entity_count"))
    assembly_count = first_number(get_nested(entry_json, "rcsb_entry_info.assembly_count"))
    deposited_model_count = first_number(get_nested(entry_json, "rcsb_entry_info.deposited_model_count"))
    deposited_atom_count = first_number(get_nested(entry_json, "rcsb_entry_info.deposited_atom_count"))
    deposited_polymer_monomer_count = first_number(get_nested(entry_json, "rcsb_entry_info.deposited_polymer_monomer_count"))
    deposited_nonpolymer_entity_count = first_number(get_nested(entry_json, "rcsb_entry_info.deposited_nonpolymer_entity_count"))

    r_free = first_number(get_nested(entry_json, "refine.0.ls_R_factor_R_free"))
    r_work = first_number(get_nested(entry_json, "refine.0.ls_R_factor_R_work"))
    r_obs = first_number(get_nested(entry_json, "refine.0.ls_R_factor_obs"))

    clashscore = first_number(
        get_nested(entry_json, "pdbx_vrpt_summary.clashscore",
                   get_nested(entry_json, "rcsb_entry_info.clashscore", np.nan))
    )

    rama_outliers = first_number(
        get_nested(entry_json, "pdbx_vrpt_summary.percent_ramachandran_outliers",
                   get_nested(entry_json, "pdbx_vrpt_summary.ramachandran_outliers", np.nan))
    )

    rotamer_outliers = first_number(
        get_nested(entry_json, "pdbx_vrpt_summary.percent_rotamer_outliers",
                   get_nested(entry_json, "pdbx_vrpt_summary.rotamer_outliers", np.nan))
    )

    rsrz_outliers = first_number(
        get_nested(entry_json, "pdbx_vrpt_summary.percent_RSRZ_outliers",
                   get_nested(entry_json, "pdbx_vrpt_summary.rsrz_outliers", np.nan))
    )

    dcc_r = first_number(get_nested(entry_json, "pdbx_vrpt_summary.DCC_R", np.nan))
    dcc_rfree = first_number(get_nested(entry_json, "pdbx_vrpt_summary.DCC_Rfree", np.nan))

    exp_method_u = exp_method.upper()
    is_xray = 1 if "X-RAY" in exp_method_u else 0
    is_cryo_em = 1 if ("ELECTRON MICROSCOPY" in exp_method_u or "CRYO-EM" in exp_method_u) else 0
    is_nmr = 1 if "NMR" in exp_method_u else 0

    atoms_per_residue = np.nan
    if deposited_atom_count > 0 and deposited_polymer_monomer_count > 0:
        atoms_per_residue = deposited_atom_count / deposited_polymer_monomer_count

    ligands_per_protein_entity = np.nan
    if protein_entity_count > 0 and deposited_nonpolymer_entity_count >= 0:
        ligands_per_protein_entity = deposited_nonpolymer_entity_count / protein_entity_count

    age_years = np.nan
    if not np.isnan(release_year):
        age_years = 2026.0 - release_year

    return {
        "pdb_id": pdb_id,
        "exp_method": exp_method,
        "is_xray": is_xray,
        "is_cryo_em": is_cryo_em,
        "is_nmr": is_nmr,
        "resolution_A": resolution,
        "deposit_year": deposit_year,
        "release_year": release_year,
        "age_years": age_years,
        "protein_entity_count": protein_entity_count,
        "total_polymer_entity_count": total_polymer_entity_count,
        "assembly_count": assembly_count,
        "deposited_model_count": deposited_model_count,
        "deposited_atom_count": deposited_atom_count,
        "deposited_polymer_monomer_count": deposited_polymer_monomer_count,
        "deposited_nonpolymer_entity_count": deposited_nonpolymer_entity_count,
        "atoms_per_residue": atoms_per_residue,
        "ligands_per_protein_entity": ligands_per_protein_entity,
        "r_free": r_free,
        "r_work": r_work,
        "r_observed": r_obs,
        "clashscore": clashscore,
        "ramachandran_outliers_pct": rama_outliers,
        "rotamer_outliers_pct": rotamer_outliers,
        "rsrz_outliers_pct": rsrz_outliers,
        "dcc_r": dcc_r,
        "dcc_rfree": dcc_rfree,
        "structure_url": f"https://files.rcsb.org/download/{pdb_id}.cif",
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def build_random_test_set(n_target: int = N_TARGET, out_csv: Path = OUT_CSV, failed_csv: Path = FAILED_CSV):
    all_ids = fetch_current_entry_ids()
    random.seed(RANDOM_SEED)
    random.shuffle(all_ids)

    rows = []
    failed = []
    used = set()

    print(f"[INFO] Total current PDB IDs: {len(all_ids)}")
    print(f"[INFO] Target successful rows: {n_target}")

    for idx, pdb_id in enumerate(all_ids, 1):
        if len(rows) >= n_target:
            break
        if pdb_id in used:
            continue
        used.add(pdb_id)

        print(f"[{idx}] Trying {pdb_id} ...")

        entry_json = safe_get_json(ENTRY_CORE_URL.format(pdb_id=pdb_id))
        if entry_json is None:
            failed.append((pdb_id, "rcsb_fetch_failed"))
            continue

        features = extract_entry_features(entry_json)
        if features is None:
            failed.append((pdb_id, "feature_extraction_failed_or_not_experimental_protein"))
            continue

        rows.append(features)
        print(f"    [OK] collected={len(rows)}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if len(rows) < n_target:
        print(f"[WARN] Requested {n_target}, but only collected {len(rows)} successful entries.")

    df = pd.DataFrame(rows)

    feature_cols = [
        "pdb_id",
        "exp_method",
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
        "structure_url",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df[feature_cols].copy()

    df.to_csv(out_csv, index=False)

    fail_df = pd.DataFrame(failed, columns=["pdb_id", "reason"])
    fail_df.to_csv(failed_csv, index=False)

    print("\n[INFO] Saved files:")
    print(f"  {out_csv}")
    print(f"  {failed_csv}")
    print(f"[INFO] Successful rows: {len(df)}")
    print(f"[INFO] Failed rows: {len(fail_df)}")

    print("\n[INFO] Preview:")
    print(df.head().to_string(index=False))

    return df


if __name__ == "__main__":
    build_random_test_set()