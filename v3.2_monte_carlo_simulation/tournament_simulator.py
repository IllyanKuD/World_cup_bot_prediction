#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 TOURNAMENT SIMULATOR - simulation Monte Carlo du reste du tournoi
==============================================================================
Simule N fois la suite du tournoi a partir de l'etat ACTUEL du bracket
(matchs deja joues = figes, matchs restants = tires au sort selon les
probabilites du modele de Poisson), en propageant les qualifications via
bracket_links.json exactement comme bracket_engine.build_effective_matches,
mais avec un resultat different a chaque tirage.

HYPOTHESES ET LIMITES (a garder en tete en lisant les resultats) :
  - En phase a elimination directe, un match nul est tranche par les tirs
    au but : on simule cela en supposant 50/50 (probabilite de victoire
    = P(victoire) + 0.5 x P(nul)). C'est une approximation raisonnable
    mais imparfaite (certaines equipes sont statistiquement meilleures
    aux tirs au but que d'autres, information non modelisee ici).
  - La "forme" de chaque equipe (utilisee par le modele pour chaque match)
    est figee a son niveau ACTUEL (calcule sur les matchs deja joues) et
    n'est PAS mise a jour au fil de la simulation d'un meme tirage, pour
    des raisons de temps de calcul. Sur un format a elimination directe
    courte (5-6 tours), l'impact de cette simplification reste limite.
  - Les resultats sont aussi bons que le modele qui les genere : a prendre
    comme un ordre de grandeur, pas une certitude.
==============================================================================
"""

import numpy as np
import pandas as pd

import bracket_engine
from fifa_2026_app import DataStore, predict_match

STAGE_LABELS = {1: "Groupes", 2: "16es", 3: "8es", 4: "Quarts",
                5: "Demies", 6: "Petite finale", 7: "Finale"}


def _match_win_prob(store: DataStore, home_id: int, away_id: int) -> float:
    """Probabilite que HOME l'emporte (nul reparti 50/50, cf tirs au but)."""
    pred = predict_match(store, home_id, away_id)
    return pred.proba_a + 0.5 * pred.proba_draw


def _qualifier(source_mid: int, kind: str, base_by_id: pd.DataFrame,
               winners: dict, losers: dict):
    """Equipe qualifiee (gagnant ou perdant) issue du match `source_mid`,
    que ce match soit deja reellement joue (base dataset) ou qu'il vienne
    d'etre simule plus tot dans CE tirage (winners/losers)."""
    source_mid = int(source_mid)
    if source_mid in winners:
        return winners[source_mid] if kind == "winner" else losers[source_mid]
    row = base_by_id.loc[source_mid]
    if row["status"] != "Completed":
        return None  # pas encore resolu (ne devrait pas arriver, ordre topologique)
    w, l = bracket_engine._winner_loser(row)
    if pd.isna(w) or pd.isna(l):
        return None
    return int(w) if kind == "winner" else int(l)


def simulate_tournament(store: DataStore, n_simulations: int = 5000, seed=None) -> dict:
    """Simule n_simulations fois le reste du tournoi.

    Retourne un dict :
      - reach_counts : {stage_id: {team_id: nb de tirages ou l'equipe a
        atteint (au moins participe a) ce stade}}
      - champion_counts : {team_id: nb de tirages ou l'equipe remporte la finale}
      - n_simulations, stages_to_report, stage_label
    ou {"error": "..."} si rien a simuler (tournoi deja termine)."""
    rng = np.random.default_rng(seed)

    links = bracket_engine.load_bracket_links(store.data_dir)
    base = store.matches.copy()
    base_by_id = base.set_index("match_id")

    pending_ids = sorted(int(x) for x in base.loc[base["status"] != "Completed", "match_id"])
    if not pending_ids:
        return {"error": "Tous les matchs sont deja termines : rien a simuler."}

    stages_to_report = sorted(int(s) for s in
                               base.loc[base["match_id"].isin(pending_ids), "stage_id"].unique())

    # equipes deja qualifiees pour un stage donne independamment de tout tirage
    # (matchs de ce stade deja Completed, OU pas encore joues mais dont les 2
    # equipes sont deja connues - ex : quart de finale deja fixe par le tirage
    # officiel avant meme que le tour precedent soit termine).
    base_reach = {sid: set() for sid in stages_to_report}
    for sid in stages_to_report:
        rows = base[(base["stage_id"] == sid)
                     & base["home_team_id"].notna() & base["away_team_id"].notna()]
        for _, r in rows.iterrows():
            base_reach[sid].add(int(r["home_team_id"]))
            base_reach[sid].add(int(r["away_team_id"]))

    reach_counts = {sid: {} for sid in stages_to_report}
    champion_counts = {}

    final_ids = base.loc[base["stage_id"] == 7, "match_id"]
    final_match_id = int(final_ids.iloc[0]) if len(final_ids) else None

    win_prob_cache = {}

    for _ in range(n_simulations):
        winners, losers = {}, {}
        reached_this_sim = {sid: set(base_reach[sid]) for sid in stages_to_report}

        for mid in pending_ids:
            row = base_by_id.loc[mid]
            h, a = row["home_team_id"], row["away_team_id"]

            if pd.isna(h) or pd.isna(a):
                link = links.get(mid)
                if link is None:
                    continue
                kind = "winner" if link["type"] == "winner" else "loser"
                h = _qualifier(link["home_source"], kind, base_by_id, winners, losers)
                a = _qualifier(link["away_source"], kind, base_by_id, winners, losers)
                if h is None or a is None:
                    continue
            h, a = int(h), int(a)

            sid = int(row["stage_id"])
            if sid in reached_this_sim:
                reached_this_sim[sid].add(h)
                reached_this_sim[sid].add(a)

            key = (h, a)
            if key not in win_prob_cache:
                win_prob_cache[key] = _match_win_prob(store, h, a)
            p_home = win_prob_cache[key]

            home_wins = rng.random() < p_home
            winner, loser = (h, a) if home_wins else (a, h)
            winners[mid], losers[mid] = winner, loser

            if mid == final_match_id:
                champion_counts[winner] = champion_counts.get(winner, 0) + 1

        for sid in stages_to_report:
            for tid in reached_this_sim[sid]:
                reach_counts[sid][tid] = reach_counts[sid].get(tid, 0) + 1

    return {
        "n_simulations": n_simulations,
        "reach_counts": reach_counts,
        "champion_counts": champion_counts,
        "stages_to_report": stages_to_report,
        "stage_label": STAGE_LABELS,
    }


def summarize_simulation(sim_result: dict, store: DataStore) -> pd.DataFrame:
    """Met en forme les resultats en DataFrame (1 ligne / equipe encore en
    course, colonnes = probabilite d'atteindre chaque stade + de gagner)."""
    if "error" in sim_result or not sim_result:
        return pd.DataFrame()

    n = sim_result["n_simulations"]
    stages = sim_result["stages_to_report"]
    label = sim_result["stage_label"]

    all_teams = set()
    for sid in stages:
        all_teams |= set(sim_result["reach_counts"][sid].keys())
    all_teams |= set(sim_result["champion_counts"].keys())

    rows = []
    for tid in all_teams:
        row = {"team_id": tid, "Equipe": store.team_name(tid)}
        for sid in stages:
            row[label[sid]] = sim_result["reach_counts"][sid].get(tid, 0) / n
        row["Vainqueur"] = sim_result["champion_counts"].get(tid, 0) / n
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = "Vainqueur" if "Vainqueur" in df.columns else label[stages[-1]]
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)
