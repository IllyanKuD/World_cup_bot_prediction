#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 BRACKET ENGINE - fusion des resultats saisis a la main + propagation
==============================================================================
Ce module NE MODIFIE JAMAIS matches.csv. Il fabrique, a la demande, une
version "effective" du calendrier en superposant :

  1) matches.csv                (les 92 matchs officiels + le calendrier)
  2) manual_results.csv         (les resultats saisis a la main - definitifs,
                                  non modifiables une fois enregistres)
  3) bracket_links.json         (qui alimente quel match suivant)

et en propageant automatiquement les equipes qualifiees dans les matchs
suivants (8emes -> quarts -> demies -> petite finale / finale) a chaque
fois qu'un match dont depend un autre est termine (que ce soit un match
du dataset original ou un match saisi a la main).

Fichier manual_results.csv (une ligne par match saisi) :
    match_id, home_score, away_score, home_penalty_score, away_penalty_score,
    home_scorers, away_scorers, entered_at

  - home_scorers / away_scorers : liste d'identifiants joueur separes par
    ";" (un id repete = un doublé/triplé), ex "12;45;12"
  - home_penalty_score / away_penalty_score : uniquement si score egal a la
    fin du temps reglementaire en phase a elimination directe.
  - La saisie est DEFINITIVE : une fois qu'un match_id est present dans ce
    fichier, il ne peut plus etre resaisi (voir save_manual_result()).
==============================================================================
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

MANUAL_RESULTS_COLUMNS = [
    "match_id", "home_score", "away_score",
    "home_penalty_score", "away_penalty_score",
    "home_scorers", "away_scorers", "entered_at",
]


def _manual_results_path(data_dir: str) -> str:
    return os.path.join(data_dir, "manual_results.csv")


def _bracket_links_path(data_dir: str) -> str:
    return os.path.join(data_dir, "bracket_links.json")


def load_manual_results(data_dir: str = ".") -> pd.DataFrame:
    """Charge manual_results.csv (retourne un DataFrame vide si absent)."""
    path = _manual_results_path(data_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=MANUAL_RESULTS_COLUMNS)
    df = pd.read_csv(path, dtype={
        "match_id": int, "home_scorers": str, "away_scorers": str,
    })
    return df


def load_bracket_links(data_dir: str = ".") -> dict:
    path = _bracket_links_path(data_dir)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items() if not k.startswith("_")}


def is_match_already_entered(match_id: int, data_dir: str = ".") -> bool:
    df = load_manual_results(data_dir)
    return int(match_id) in df["match_id"].astype(int).values if len(df) else False


def save_manual_result(match_id: int, home_score: int, away_score: int,
                        home_scorers: list, away_scorers: list,
                        home_penalty_score=None, away_penalty_score=None,
                        data_dir: str = ".") -> tuple:
    """Enregistre un resultat saisi a la main. Definitif : refuse si le
    match_id existe deja dans manual_results.csv.
    Retourne (ok: bool, message: str)."""
    df = load_manual_results(data_dir)
    if len(df) and int(match_id) in df["match_id"].astype(int).values:
        return False, (f"Le match #{match_id} a deja ete saisi. "
                        "La saisie est definitive, elle ne peut pas etre modifiee.")

    row = {
        "match_id": int(match_id),
        "home_score": int(home_score),
        "away_score": int(away_score),
        "home_penalty_score": "" if home_penalty_score is None else int(home_penalty_score),
        "away_penalty_score": "" if away_penalty_score is None else int(away_penalty_score),
        "home_scorers": ";".join(str(p) for p in home_scorers),
        "away_scorers": ";".join(str(p) for p in away_scorers),
        "entered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(_manual_results_path(data_dir), index=False)
    return True, f"Match #{match_id} enregistre."


def _winner_loser(row) -> tuple:
    """Determine (winner_team_id, loser_team_id) d'un match termine,
    en utilisant les tirs au but en cas d'egalite."""
    hs, as_ = row["home_score"], row["away_score"]
    if hs > as_:
        return row["home_team_id"], row["away_team_id"]
    if as_ > hs:
        return row["away_team_id"], row["home_team_id"]
    # egalite -> tirs au but obligatoires (phase a elimination directe)
    hp, ap = row.get("home_penalty_score"), row.get("away_penalty_score")
    if pd.notna(hp) and pd.notna(ap) and hp != ap:
        return (row["home_team_id"], row["away_team_id"]) if hp > ap \
            else (row["away_team_id"], row["home_team_id"])
    return None, None  # pas assez d'info pour trancher


def build_effective_matches(matches_df: pd.DataFrame, data_dir: str = "."):
    """Fusionne matches.csv + manual_results.csv, puis propage les
    qualifications dans le bracket autant de fois que necessaire.

    Retourne (effective_df, log) ou log est une liste de messages
    decrivant les propagations effectuees (utile pour affichage debug/UI)."""
    eff = matches_df.copy()
    eff["source"] = "dataset"

    manual = load_manual_results(data_dir)
    for _, mr in manual.iterrows():
        mid = int(mr["match_id"])
        mask = eff["match_id"] == mid
        if not mask.any():
            continue
        eff.loc[mask, "home_score"] = mr["home_score"]
        eff.loc[mask, "away_score"] = mr["away_score"]
        if str(mr.get("home_penalty_score", "")) not in ("", "nan"):
            eff.loc[mask, "home_penalty_score"] = mr["home_penalty_score"]
            eff.loc[mask, "away_penalty_score"] = mr["away_penalty_score"]
        eff.loc[mask, "status"] = "Completed"
        eff.loc[mask, "source"] = "manual"

    links = load_bracket_links(data_dir)
    log = []
    # plusieurs passes : une propagation peut en debloquer une autre
    # (ex: 8eme -> quart -> demie dans le meme rafraichissement)
    for _ in range(len(links) + 1):
        changed = False
        for target_id, link in links.items():
            trow = eff.loc[eff["match_id"] == target_id]
            if len(trow) == 0:
                continue
            trow = trow.iloc[0]
            if pd.notna(trow["home_team_id"]) and pd.notna(trow["away_team_id"]):
                continue  # deja rempli

            home_src = eff.loc[eff["match_id"] == link["home_source"]]
            away_src = eff.loc[eff["match_id"] == link["away_source"]]
            if len(home_src) == 0 or len(away_src) == 0:
                continue
            home_src, away_src = home_src.iloc[0], away_src.iloc[0]
            if home_src["status"] != "Completed" or away_src["status"] != "Completed":
                continue

            hw, hl = _winner_loser(home_src)
            aw, al = _winner_loser(away_src)
            if link["type"] == "winner":
                home_team, away_team = hw, aw
            else:  # "loser" -> petite finale
                home_team, away_team = hl, al
            if home_team is None or away_team is None:
                continue

            idx = eff.index[eff["match_id"] == target_id][0]
            eff.loc[idx, "home_team_id"] = home_team
            eff.loc[idx, "away_team_id"] = away_team
            log.append(
                f"Match #{target_id} : equipes qualifiees mises a jour "
                f"({int(home_team)} vs {int(away_team)})."
            )
            changed = True
        if not changed:
            break

    return eff, log


def build_effective_events(events_df: pd.DataFrame, data_dir: str = "."):
    """Ajoute au dataframe d'evenements les buts saisis a la main (pour que
    le detail de match et le top buteurs les integrent naturellement)."""
    manual = load_manual_results(data_dir)
    if len(manual) == 0:
        return events_df.copy()

    extra_rows = []
    for _, mr in manual.iterrows():
        mid = int(mr["match_id"])
        for side, col in (("home", "home_scorers"), ("away", "away_scorers")):
            raw = mr.get(col, "")
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            for pid in str(raw).split(";"):
                if pid == "":
                    continue
                extra_rows.append({
                    "match_id": mid, "minute": None,
                    "event_type": "Goal", "player_id": int(pid),
                    "team_id": None,  # rempli par l'appelant (a besoin de squads)
                    "_side": side,
                })
    if not extra_rows:
        return events_df.copy()
    extra_df = pd.DataFrame(extra_rows)
    return pd.concat([events_df, extra_df], ignore_index=True)


def pending_playable_matches(effective_df: pd.DataFrame, data_dir: str = "."):
    """Retourne les matchs dont les 2 equipes sont connues, qui ne sont pas
    encore Completed, et qui n'ont pas encore ete saisis manuellement :
    ce sont les matchs a proposer dans l'onglet de saisie."""
    manual_ids = set(load_manual_results(data_dir)["match_id"].astype(int)) \
        if len(load_manual_results(data_dir)) else set()
    df = effective_df[
        (effective_df["status"] != "Completed")
        & effective_df["home_team_id"].notna()
        & effective_df["away_team_id"].notna()
        & (~effective_df["match_id"].isin(manual_ids))
    ]
    return df.sort_values("match_id")
