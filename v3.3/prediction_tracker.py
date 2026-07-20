#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 PREDICTION TRACKER - fiabilite de l'IA face au reel (v3)
==============================================================================
Pour chaque match reellement joue (dataset original OU saisi a la main),
on reconstruit la prediction que l'IA aurait faite EN NE SE SERVANT QUE DES
MATCHS ANTERIEURS a la date de ce match ("walk-forward evaluation") :
un modele different peut etre re-entraine pour chaque match evalue, ce qui
evite toute fuite de donnees venant du futur (y compris implicitement via
les coefficients du modele).

Deux statistiques separees sont calculees (comme demande) :

1) RESULTAT (Victoire/Nul/Defaite) : le modele de regression logistique
   classe les 3 issues par probabilite. On regarde si l'issue reelle etait
   son choix n1, n2 ou n3.

2) SCORE EXACT : le modele de Poisson classe tous les scores possibles par
   probabilite. On regarde le rang du score reellement observe dans ce
   classement (top1 / top2 / top3 / au-dela).

Necessite un minimum de matchs anterieurs pour etre evalue (MIN_TRAIN_MATCHES) ;
les tout premiers matchs du tournoi sont donc exclus (pas assez d'historique
pour qu'un modele entraine "a l'epoque" ait un sens).
==============================================================================
"""

import numpy as np
import pandas as pd

from fifa_2026_app import DataStore, poisson_pmf

MIN_TRAIN_MATCHES = 20
MAX_GOALS = 6

STAKE_DEFAULT = 10.0  # mise de reference pour les tableaux "si on avait mis 10E"

# ==========================================================================
# PARIS SPORTIFS - cotes "equitables" (implied odds, sans marge bookmaker)
# ==========================================================================
# Le dataset ne contient aucune cote de bookmaker : les "cotes" ci-dessous
# sont deduites des probabilites des modeles (cote = 1 / probabilite).
# Ce sont donc des cotes theoriques ("fair odds"), pas des cotes reelles
# de bookmaker (qui incluraient une marge). Elles servent a comparer les
# 2 modeles de facon homogene.
SINGLE_BET_FROM_OUTCOME = {"H": "1", "D": "N", "A": "2"}
OUTCOME_FROM_SINGLE_BET = {v: k for k, v in SINGLE_BET_FROM_OUTCOME.items()}
# double chance -> quelle issue est EXCLUE par ce pari
DOUBLE_CHANCE_EXCLUDES = {"1X": "A", "12": "D", "X2": "H"}
# quelle issue, une fois exclue, mene a quel pari double chance
DOUBLE_CHANCE_FROM_EXCLUDED = {v: k for k, v in DOUBLE_CHANCE_EXCLUDES.items()}


def _outcome_probs_from_score_grid(score_probs: dict) -> dict:
    """Somme la matrice de scores du modele de Poisson pour obtenir ses
    probabilites implicites de Victoire/Nul/Defaite (H/D/A)."""
    pH = pD = pA = 0.0
    for (i, j), p in score_probs.items():
        if i > j:
            pH += p
        elif i == j:
            pD += p
        else:
            pA += p
    return {"H": pH, "D": pD, "A": pA}


def _fair_odds(p: float) -> float:
    """Cote 'equitable' = 1 / probabilite (pas de marge bookmaker)."""
    return float("inf") if p <= 0 else 1.0 / p


def market_odds(outcome_probs: dict) -> dict:
    """A partir de probabilites {'H','D','A'}, calcule les 6 cotes du
    marche 1N2 + double chance : 1, N, 2, 1X, X2, 12."""
    pH, pD, pA = outcome_probs["H"], outcome_probs["D"], outcome_probs["A"]
    return {
        "1": _fair_odds(pH), "N": _fair_odds(pD), "2": _fair_odds(pA),
        "1X": _fair_odds(pH + pD), "X2": _fair_odds(pD + pA), "12": _fair_odds(pH + pA),
    }


def ai_picks(outcome_probs: dict) -> tuple:
    """Determine les 2 choix de l'IA a partir de ses probabilites H/D/A :
      - pick1  : l'issue la plus probable ('1', 'N' ou '2')
      - pick_dc: le pari double chance qui couvre les 2 issues les plus
                 probables (equivalent a exclure la moins probable)."""
    ranked = sorted(outcome_probs.items(), key=lambda kv: kv[1], reverse=True)
    best_outcome = ranked[0][0]
    least_likely_outcome = ranked[2][0]
    pick1 = SINGLE_BET_FROM_OUTCOME[best_outcome]
    pick_dc = DOUBLE_CHANCE_FROM_EXCLUDED[least_likely_outcome]
    return pick1, pick_dc


def bet_wins(pick_code: str, actual_outcome: str) -> bool:
    """Le pari 'pick_code' (ex '1', 'X2'...) est-il gagnant sachant l'issue
    reelle du match (H/D/A) ?"""
    if pick_code in OUTCOME_FROM_SINGLE_BET:
        return OUTCOME_FROM_SINGLE_BET[pick_code] == actual_outcome
    return DOUBLE_CHANCE_EXCLUDES[pick_code] != actual_outcome


def _team_form_asof(long_df: pd.DataFrame, team_id, date):
    rows = long_df[(long_df["team_id"] == team_id) & (long_df["date"] < date)]
    if len(rows) == 0:
        return dict(
            goals_for=long_df["goals_for"].mean(),
            goals_against=long_df["goals_against"].mean(),
            xg_for=long_df["xg_for"].mean(),
            xg_against=long_df["xg_against"].mean(),
            sot=long_df["shots_on_target"].mean(),
        )
    return dict(
        goals_for=rows["goals_for"].mean(),
        goals_against=rows["goals_against"].mean(),
        xg_for=rows["xg_for"].mean(),
        xg_against=rows["xg_against"].mean(),
        sot=rows["shots_on_target"].mean(),
    )


def _feature_row(store, long_df, team_id, opp_id, is_home_flag, date):
    form = _team_form_asof(long_df, team_id, date)
    elo_diff = store.teams.loc[team_id, "elo_rating"] - store.teams.loc[opp_id, "elo_rating"]
    rank_diff = (store.teams.loc[opp_id, "fifa_ranking_pre_tournament"]
                 - store.teams.loc[team_id, "fifa_ranking_pre_tournament"])
    val_a = store._squad_agg.loc[team_id, "avg_value"] if team_id in store._squad_agg.index else 0
    val_b = store._squad_agg.loc[opp_id, "avg_value"] if opp_id in store._squad_agg.index else 0
    value_diff = np.log1p(val_a) - np.log1p(val_b)
    is_host = store.teams.loc[team_id, "is_host"]
    return [is_home_flag, elo_diff, rank_diff, value_diff, is_host,
            form["goals_for"], form["goals_against"],
            form["xg_for"], form["xg_against"], form["sot"]]


def evaluate_predictions(store: DataStore, min_train_matches: int = MIN_TRAIN_MATCHES,
                          max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """Retourne un DataFrame, une ligne par match evalue, avec le detail de
    la prediction retrospective et sa comparaison au resultat reel."""
    completed = store.matches[store.matches["status"] == "Completed"].sort_values("date").copy()
    long_df = store._long

    records = []
    for _, m in completed.iterrows():
        date_m = m["date"]
        hid, aid = m["home_team_id"], m["away_team_id"]
        if pd.isna(hid) or pd.isna(aid):
            continue

        train_long = long_df[long_df["date"] < date_m]
        train_feat = store.features[store.features["date"] < date_m]
        if len(train_long) < min_train_matches or len(train_feat) < min_train_matches:
            continue  # pas assez d'historique pour entrainer un modele "de l'epoque"

        # -- re-entrainement "a l'epoque" (walk-forward) --
        try:
            poisson_model, scaler, cols = DataStore.fit_poisson(train_long, DataStore.FEATURE_COLS)
            logit_model = DataStore.fit_logit(train_feat)
        except Exception:
            continue

        row_h = _feature_row(store, long_df, hid, aid, 1, date_m)
        row_a = _feature_row(store, long_df, aid, hid, 0, date_m)
        Xs = scaler.transform(np.array([row_h, row_a]))
        lam_h, lam_a = poisson_model.predict(Xs)
        lam_h, lam_a = max(lam_h, 0.05), max(lam_a, 0.05)

        score_probs = {}
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                score_probs[(i, j)] = poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
        ranked_scores = sorted(score_probs.items(), key=lambda kv: kv[1], reverse=True)

        elo_diff = store.teams.loc[hid, "elo_rating"] - store.teams.loc[aid, "elo_rating"]
        rank_diff = (store.teams.loc[aid, "fifa_ranking_pre_tournament"]
                     - store.teams.loc[hid, "fifa_ranking_pre_tournament"])
        val_h = store._squad_agg.loc[hid, "avg_value"] if hid in store._squad_agg.index else 0
        val_a = store._squad_agg.loc[aid, "avg_value"] if aid in store._squad_agg.index else 0
        value_diff = np.log1p(val_h) - np.log1p(val_a)
        host_diff = store.teams.loc[hid, "is_host"] - store.teams.loc[aid, "is_host"]
        logit_X = np.array([[elo_diff, rank_diff, value_diff, host_diff]])
        outcome_probs = dict(zip(logit_model.classes_, logit_model.predict_proba(logit_X)[0]))
        ranked_outcomes = sorted(outcome_probs.items(), key=lambda kv: kv[1], reverse=True)

        actual_score = (int(m["home_score"]), int(m["away_score"]))
        actual_outcome = ("H" if m["home_score"] > m["away_score"]
                           else ("A" if m["away_score"] > m["home_score"] else "D"))

        score_rank = next((k + 1 for k, (sc, _) in enumerate(ranked_scores)
                            if sc == actual_score), None)
        outcome_rank = next((k + 1 for k, (oc, _) in enumerate(ranked_outcomes)
                              if oc == actual_outcome), None)

        poisson_outcome_probs = _outcome_probs_from_score_grid(score_probs)
        odds_by_model = {
            "poisson": market_odds(poisson_outcome_probs),
            "logit": market_odds(outcome_probs),
        }
        probs_by_model = {"poisson": poisson_outcome_probs, "logit": outcome_probs}

        betting_cols = {}
        for model_key, odds in odds_by_model.items():
            pick1, pick_dc = ai_picks(probs_by_model[model_key])
            for bet_kind, pick in (("pick1", pick1), ("pickdc", pick_dc)):
                betting_cols[f"{model_key}_{bet_kind}_bet"] = pick
                betting_cols[f"{model_key}_{bet_kind}_odds"] = odds[pick]
                betting_cols[f"{model_key}_{bet_kind}_win"] = bet_wins(pick, actual_outcome)
            for code, val in odds.items():
                betting_cols[f"{model_key}_odds_{code}"] = val

        records.append({
            "match_id": int(m["match_id"]), "date": date_m,
            "stage_id": m["stage_id"],
            "home_team": store.team_name(hid), "away_team": store.team_name(aid),
            "actual_score": f"{actual_score[0]}-{actual_score[1]}",
            "actual_outcome": actual_outcome,
            "predicted_best_score": f"{ranked_scores[0][0][0]}-{ranked_scores[0][0][1]}",
            "predicted_best_outcome": ranked_outcomes[0][0],
            "score_rank": score_rank,
            "outcome_rank": outcome_rank,
            "outcome_proba_of_actual": outcome_probs.get(actual_outcome, 0.0),
            "score_proba_of_actual": score_probs.get(actual_score, 0.0),
            "source": m.get("source", "dataset"),
            "logit_proba_H": outcome_probs.get("H", 0.0),
            "logit_proba_D": outcome_probs.get("D", 0.0),
            "logit_proba_A": outcome_probs.get("A", 0.0),
            "poisson_proba_H": poisson_outcome_probs["H"],
            "poisson_proba_D": poisson_outcome_probs["D"],
            "poisson_proba_A": poisson_outcome_probs["A"],
            **betting_cols,
        })

    return pd.DataFrame(records)


def summarize(evals: pd.DataFrame) -> dict:
    """Calcule les statistiques agregees a partir du DataFrame d'evaluation."""
    if len(evals) == 0:
        return {}

    n = len(evals)
    outcome_top1 = (evals["outcome_rank"] == 1).mean()
    outcome_top2 = (evals["outcome_rank"] <= 2).mean()

    def score_topk(k):
        return (evals["score_rank"].fillna(99) <= k).mean()

    stage_names = {1: "Groupes", 2: "16es", 3: "8es", 4: "Quarts",
                   5: "Demies", 6: "Petite finale", 7: "Finale"}
    by_stage = evals.copy()
    by_stage["stage_name"] = by_stage["stage_id"].map(stage_names)
    stage_outcome = by_stage.groupby("stage_name")["outcome_rank"].apply(
        lambda s: (s == 1).mean()
    )

    biggest_upsets = evals.sort_values("outcome_proba_of_actual").head(5)

    return {
        "n_matches_evaluated": n,
        "outcome_top1_pct": outcome_top1,
        "outcome_top2_pct": outcome_top2,
        "score_top1_pct": score_topk(1),
        "score_top2_pct": score_topk(2),
        "score_top3_pct": score_topk(3),
        "avg_proba_actual_outcome": evals["outcome_proba_of_actual"].mean(),
        "avg_proba_actual_score": evals["score_proba_of_actual"].mean(),
        "accuracy_by_stage": stage_outcome,
        "biggest_upsets": biggest_upsets,
    }


# ==========================================================================
# PARIS SPORTIFS - recapitulatifs
# ==========================================================================
MODEL_LABELS = {"poisson": "Modele Poisson (score exact)",
                "logit": "Modele logistique (issue)"}
BET_KIND_LABELS = {"pick1": "Choix 1 (issue la plus probable)",
                    "pickdc": "Choix 1+2 (double chance)"}


def _betting_rows(evals: pd.DataFrame, bet_kind: str, stake: float) -> pd.DataFrame:
    """Une ligne par modele (poisson/logit) : bilan si on avait joue
    'bet_kind' ('pick1' ou 'pickdc') sur tous les matchs evalues, avec une
    mise fixe de `stake` par match."""
    rows = []
    for model_key, label in MODEL_LABELS.items():
        odds_col = f"{model_key}_{bet_kind}_odds"
        win_col = f"{model_key}_{bet_kind}_win"
        bet_col = f"{model_key}_{bet_kind}_bet"
        n = len(evals)
        n_win = int(evals[win_col].sum())
        total_staked = n * stake
        total_returned = float((evals[win_col].astype(int) * evals[odds_col] * stake).sum())
        profit = total_returned - total_staked
        roi_pct = (profit / total_staked * 100) if total_staked else 0.0
        avg_odds_played = float(evals[odds_col].mean())
        avg_odds_won = float(evals.loc[evals[win_col], odds_col].mean()) if n_win else 0.0
        rows.append({
            "Modele": label,
            "Paris joues": n,
            "Paris gagnes": n_win,
            "Taux de reussite": n_win / n if n else 0.0,
            "Cote moyenne jouee": avg_odds_played,
            "Cote moyenne (paris gagnes)": avg_odds_won,
            "Mise totale (EUR)": total_staked,
            "Total retourne (EUR)": total_returned,
            "Profit / Perte (EUR)": profit,
            "ROI (%)": roi_pct,
            "_bet_col": bet_col,
        })
    return pd.DataFrame(rows).drop(columns="_bet_col")


def summarize_betting(evals: pd.DataFrame, stake: float = STAKE_DEFAULT) -> dict:
    """Construit les tableaux recapitulatifs de paris sportifs a partir du
    DataFrame retourne par evaluate_predictions (qui contient deja les
    cotes/choix des 2 modeles, calcules par match en walk-forward).

    Retourne un dict avec :
      - recap_pick1   : DataFrame (1 ligne / modele) - bilan "Choix 1"
      - recap_pickdc  : DataFrame (1 ligne / modele) - bilan "Choix 1+2"
      - detail        : DataFrame match par match, toutes colonnes de paris
      - stake         : la mise utilisee pour les totaux en euros
    """
    if len(evals) == 0:
        return {}

    recap_pick1 = _betting_rows(evals, "pick1", stake)
    recap_pickdc = _betting_rows(evals, "pickdc", stake)

    detail_cols = [
        "match_id", "date", "stage_id", "home_team", "away_team",
        "actual_score", "actual_outcome", "source",
        "poisson_pick1_bet", "poisson_pick1_odds", "poisson_pick1_win",
        "poisson_pickdc_bet", "poisson_pickdc_odds", "poisson_pickdc_win",
        "logit_pick1_bet", "logit_pick1_odds", "logit_pick1_win",
        "logit_pickdc_bet", "logit_pickdc_odds", "logit_pickdc_win",
        "poisson_odds_1", "poisson_odds_N", "poisson_odds_2",
        "poisson_odds_1X", "poisson_odds_X2", "poisson_odds_12",
        "logit_odds_1", "logit_odds_N", "logit_odds_2",
        "logit_odds_1X", "logit_odds_X2", "logit_odds_12",
    ]
    detail = evals[[c for c in detail_cols if c in evals.columns]].copy()

    return {
        "recap_pick1": recap_pick1,
        "recap_pickdc": recap_pickdc,
        "detail": detail,
        "stake": stake,
    }


# ==========================================================================
# CALIBRATION DES MODELES - diagramme de fiabilite + score de Brier
# ==========================================================================
# Un modele est "bien calibre" si, parmi tous les cas ou il annonce une
# probabilite de ~70%, l'evenement se produit reellement environ 70% du
# temps (ni plus, ni moins). On traite les 3 issues (H/D/A) comme 3
# evenements binaires independants ("one-vs-rest"), regroupes ensemble :
# c'est la methode standard pour tracer une courbe de calibration
# multiclasse et calculer un score de Brier multiclasse.
OUTCOME_CODES = ("H", "D", "A")


def _pooled_proba_actual(evals: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Aplati le DataFrame d'evaluation : une ligne par (match, issue
    possible), avec la probabilite annoncee par le modele pour cette issue
    et un indicateur binaire (1 si c'est effectivement l'issue reelle)."""
    rows = []
    for _, r in evals.iterrows():
        for code in OUTCOME_CODES:
            rows.append({
                "match_id": r["match_id"],
                "outcome_code": code,
                "predicted_proba": r[f"{model_key}_proba_{code}"],
                "actual": 1 if r["actual_outcome"] == code else 0,
            })
    return pd.DataFrame(rows)


def calibration_curve(evals: pd.DataFrame, model_key: str, n_bins: int = 10) -> pd.DataFrame:
    """Diagramme de fiabilite (reliability diagram) pour un modele
    ('poisson' ou 'logit') : regroupe les probabilites annoncees en
    `n_bins` tranches egales et compare, pour chaque tranche, la
    probabilite moyenne annoncee a la frequence reellement observee.

    Un modele parfaitement calibre aurait 'proba_moyenne_annoncee' ==
    'frequence_observee' sur toute la courbe (la diagonale y = x)."""
    flat = _pooled_proba_actual(evals, model_key)
    if len(flat) == 0:
        return pd.DataFrame()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    flat["bin"] = pd.cut(flat["predicted_proba"], bins=bins, include_lowest=True)

    agg = flat.groupby("bin", observed=True).agg(
        proba_moyenne_annoncee=("predicted_proba", "mean"),
        frequence_observee=("actual", "mean"),
        n_observations=("actual", "size"),
    ).reset_index()
    agg["ecart"] = agg["proba_moyenne_annoncee"] - agg["frequence_observee"]
    agg["bin_label"] = agg["bin"].apply(
        lambda b: f"{b.left:.0%}-{b.right:.0%}" if pd.notna(b) else ""
    )
    return agg


def brier_score(evals: pd.DataFrame, model_key: str) -> float:
    """Score de Brier multiclasse (H/D/A) : moyenne des erreurs quadratiques
    entre les probabilites annoncees et le resultat reel (encode en
    'one-hot' sur les 3 issues). Plus BAS = meilleur. 0 = parfait,
    l'ordre de grandeur habituel pour un pronostic 1N2 raisonnable se situe
    autour de 0.55-0.65 (une prediction 'ignorante' qui donnerait toujours
    33/33/33% obtient environ 0.667)."""
    flat = _pooled_proba_actual(evals, model_key)
    if len(flat) == 0:
        return float("nan")
    return float(((flat["predicted_proba"] - flat["actual"]) ** 2).mean() * len(OUTCOME_CODES))


def calibration_summary(evals: pd.DataFrame, n_bins: int = 10) -> dict:
    """Calibration + Brier pour les 2 modeles, prets a etre affiches."""
    if len(evals) == 0:
        return {}
    return {
        "poisson": {
            "curve": calibration_curve(evals, "poisson", n_bins),
            "brier": brier_score(evals, "poisson"),
        },
        "logit": {
            "curve": calibration_curve(evals, "logit", n_bins),
            "brier": brier_score(evals, "logit"),
        },
        "n_matches": len(evals),
    }
