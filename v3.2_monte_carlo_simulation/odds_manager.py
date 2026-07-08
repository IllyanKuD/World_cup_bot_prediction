#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 ODDS MANAGER - cotes REELLES saisies a la main (source : OddsPortal)
==============================================================================
OddsPortal interdit le scraping automatise dans ses CGU et protege son site
(Cloudflare, contenu charge en JS) : il est donc impossible - et non
souhaitable - d'y brancher un robot depuis cette application. La demarche
retenue ici est la saisie manuelle : l'utilisateur consulte oddsportal.com,
releve les cotes fermees (closing odds) du marche 1N2, et eventuellement
celles du marche Double Chance, puis les colle dans le formulaire dedie du
tableau de bord.

Fichier manual_odds.csv (une ligne par match, mise a jour possible - a la
difference de manual_results.csv, une cote n'est pas "definitive") :
    match_id, bookmaker, odds_1, odds_N, odds_2,
    odds_1X, odds_X2, odds_12, source, entered_at

  - odds_1 / odds_N / odds_2 : cotes decimales du marche 1N2 (obligatoires).
  - odds_1X / odds_X2 / odds_12 : cotes decimales du marche Double Chance
    (facultatives). Si absentes, elles sont estimees a partir des cotes 1N2
    via la formule actuarielle standard de combinaison de 2 issues
    mutuellement exclusives : cote(A ou B) = 1 / (1/cote_A + 1/cote_B).
    C'est une approximation (elle ignore que le bookmaker applique en
    general une marge differente sur le marche double chance) : si tu as
    les vraies cotes double chance sous les yeux sur OddsPortal, saisis-les
    directement pour plus de precision.
==============================================================================
"""

import os
from datetime import datetime, timezone

import pandas as pd

MANUAL_ODDS_COLUMNS = [
    "match_id", "bookmaker", "odds_1", "odds_N", "odds_2",
    "odds_1X", "odds_X2", "odds_12", "source", "entered_at",
]

DC_FROM_SINGLE = {"1X": ("odds_1", "odds_N"), "X2": ("odds_N", "odds_2"), "12": ("odds_1", "odds_2")}


def _manual_odds_path(data_dir: str) -> str:
    return os.path.join(data_dir, "manual_odds.csv")


def load_manual_odds(data_dir: str = ".") -> pd.DataFrame:
    """Charge manual_odds.csv (DataFrame vide si le fichier n'existe pas)."""
    path = _manual_odds_path(data_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=MANUAL_ODDS_COLUMNS)
    df = pd.read_csv(path)
    if "match_id" in df.columns:
        df["match_id"] = df["match_id"].astype(int)
    return df


def get_odds_for_match(match_id: int, data_dir: str = "."):
    """Retourne la ligne de cotes d'un match (Series) ou None si absente."""
    df = load_manual_odds(data_dir)
    row = df[df["match_id"] == int(match_id)]
    return row.iloc[0] if len(row) else None


def save_manual_odds(match_id: int, odds_1: float, odds_N: float, odds_2: float,
                      odds_1X=None, odds_X2=None, odds_12=None,
                      bookmaker: str = "", source: str = "OddsPortal",
                      data_dir: str = ".") -> tuple:
    """Enregistre (ou met a jour) les cotes reelles d'un match. Contrairement
    aux resultats, une cote peut etre resaisie/corrigee : upsert sur
    match_id. Retourne (ok: bool, message: str)."""
    if odds_1 <= 1 or odds_N <= 1 or odds_2 <= 1:
        return False, "Les cotes decimales doivent etre strictement superieures a 1."

    if odds_1X in (None, 0):
        odds_1X = 1.0 / (1.0 / odds_1 + 1.0 / odds_N)
    if odds_X2 in (None, 0):
        odds_X2 = 1.0 / (1.0 / odds_N + 1.0 / odds_2)
    if odds_12 in (None, 0):
        odds_12 = 1.0 / (1.0 / odds_1 + 1.0 / odds_2)

    df = load_manual_odds(data_dir)
    df = df[df["match_id"] != int(match_id)]  # supprime l'ancienne saisie si existante

    row = {
        "match_id": int(match_id), "bookmaker": bookmaker,
        "odds_1": float(odds_1), "odds_N": float(odds_N), "odds_2": float(odds_2),
        "odds_1X": float(odds_1X), "odds_X2": float(odds_X2), "odds_12": float(odds_12),
        "source": source,
        "entered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(_manual_odds_path(data_dir), index=False)
    return True, f"Cotes du match #{match_id} enregistrees."


def delete_manual_odds(match_id: int, data_dir: str = ".") -> tuple:
    df = load_manual_odds(data_dir)
    if not len(df) or int(match_id) not in df["match_id"].values:
        return False, f"Aucune cote saisie pour le match #{match_id}."
    df = df[df["match_id"] != int(match_id)]
    df.to_csv(_manual_odds_path(data_dir), index=False)
    return True, f"Cotes du match #{match_id} supprimees."


def matches_missing_real_odds(evals: pd.DataFrame, data_dir: str = ".") -> pd.DataFrame:
    """Parmi les matchs deja evalues (walk-forward, cf prediction_tracker),
    lesquels n'ont pas encore de cotes reelles saisies."""
    odds_df = load_manual_odds(data_dir)
    have_odds = set(odds_df["match_id"]) if len(odds_df) else set()
    return evals[~evals["match_id"].isin(have_odds)].sort_values("date")


BET_TO_ODDS_COL = {"1": "odds_1", "N": "odds_N", "2": "odds_2",
                    "1X": "odds_1X", "X2": "odds_X2", "12": "odds_12"}


def real_odds_betting_table(evals: pd.DataFrame, model_key: str, bet_kind: str,
                             data_dir: str = ".") -> pd.DataFrame:
    """Pour un modele ('poisson'/'logit') et une strategie ('pick1'/'pickdc'),
    reprend le pari que l'IA aurait joue sur chaque match (deja calcule dans
    `evals` par prediction_tracker.evaluate_predictions) et l'evalue avec
    la VRAIE cote saisie manuellement (au lieu de la cote equitable du
    modele). Ne garde que les matchs pour lesquels une cote reelle existe."""
    odds_df = load_manual_odds(data_dir)
    if len(odds_df) == 0:
        return evals.iloc[0:0].copy()

    bet_col = f"{model_key}_{bet_kind}_bet"
    merged = evals[["match_id", "date", "home_team", "away_team",
                     "actual_score", "actual_outcome", "source", bet_col]].merge(
        odds_df, on="match_id", how="inner"
    )
    if len(merged) == 0:
        return merged

    def _real_odds(row):
        col = BET_TO_ODDS_COL[row[bet_col]]
        return row[col]

    merged["real_odds_played"] = merged.apply(_real_odds, axis=1)
    return merged


def summarize_real_odds_betting(evals: pd.DataFrame, model_key: str, bet_kind: str,
                                 stake: float, data_dir: str = ".") -> dict:
    """Bilan (paris joues/gagnes, mise, retour, profit, ROI) pour un modele
    et une strategie donnes, evalue avec les vraies cotes saisies."""
    import prediction_tracker as pt  # import local pour eviter le cycle au chargement du module

    tbl = real_odds_betting_table(evals, model_key, bet_kind, data_dir=data_dir)
    if len(tbl) == 0:
        return {"n_bets": 0}

    tbl["win"] = tbl.apply(
        lambda r: pt.bet_wins(r[f"{model_key}_{bet_kind}_bet"], r["actual_outcome"]), axis=1
    )
    n = len(tbl)
    n_win = int(tbl["win"].sum())
    total_staked = n * stake
    total_returned = float((tbl["win"].astype(int) * tbl["real_odds_played"] * stake).sum())
    profit = total_returned - total_staked
    roi_pct = (profit / total_staked * 100) if total_staked else 0.0

    return {
        "n_bets": n, "n_win": n_win,
        "win_rate": n_win / n if n else 0.0,
        "avg_odds_played": float(tbl["real_odds_played"].mean()),
        "avg_odds_won": float(tbl.loc[tbl["win"], "real_odds_played"].mean()) if n_win else 0.0,
        "total_staked": total_staked, "total_returned": total_returned,
        "profit": profit, "roi_pct": roi_pct,
        "detail": tbl,
    }
