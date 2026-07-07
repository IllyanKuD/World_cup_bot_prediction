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
