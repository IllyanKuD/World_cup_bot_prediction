#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 FIFA WORLD CUP 2026 - ANALYTICS & PREDICTION ENGINE (v2)
==============================================================================

Ce programme exploite l'integralite du dataset relationnel fourni :
  teams.csv, venues.csv, referees.csv, tournament_stages.csv,
  matches.csv / matches_detailed.csv, match_team_stats.csv,
  match_events.csv, match_lineups.csv,
  squads_and_players.csv, player_stats.csv,
  match_prediction_features.csv

Contrairement a la version 1 (qui ne disposait que de stats agregees par
joueur sans aucune date), ce dataset contient les VRAIS matchs joues avec
leurs dates : il est donc desormais possible d'avoir de vraies statistiques
"par jour", le detail de chaque match (buts, xG, cartons, compositions,
timeline minute par minute) et un moteur de prediction entraine sur des
donnees reelles plutot qu'une simple regle de trois.

------------------------------------------------------------------------
MOTEUR DE PREDICTION - methodologie
------------------------------------------------------------------------
Deux modeles complementaires, entraines sur les 92 matchs deja joues :

1) REGRESSION DE POISSON (buts attendus, sklearn PoissonRegressor)
   Les buts marques par une equipe suivent (approximativement) une loi
   de Poisson. On met les 92 matchs en "format long" (2 lignes par match :
   une par equipe), soit 184 observations, et on modelise :

       buts_marques ~ f(domicile, ecart_elo, ecart_classement_FIFA,
                        ecart_valeur_marchande, statut_pays_hote,
                        forme_recente_attaque, forme_recente_defense,
                        xG_recent_pour, xG_recent_contre, tirs_cadres_recents)

   "Forme recente" = moyenne CUMULEE des matchs precedents de l'equipe
   (calculee sans fuite de donnees : uniquement les matchs anterieurs a
   celui predit, via un shift().expanding().mean()).

   On obtient ainsi un lambda (nombre de buts attendus) pour chaque
   equipe, puis on construit la matrice de probabilite de chaque score
   exact via la loi de Poisson - exactement comme le modele historique
   de Dixon-Coles utilise en analytics football.

2) REGRESSION LOGISTIQUE MULTINOMIALE (issue du match : Victoire/Nul/Defaite)
   Entrainee directement sur match_prediction_features.csv (les memes
   92 matchs), avec les ecarts elo/classement/valeur/statut hote comme
   variables explicatives. Sert de verification croisee independante des
   probabilites issues du modele de Poisson.

Les deux resultats sont affiches cote a cote : si les deux methodes sont
d'accord, la prediction est plus fiable ; si elles divergent, cela indique
un match plus incertain qu'il n'y parait.

LIMITE IMPORTANTE : 92 matchs est un echantillon modeste pour un modele
statistique (le football reste un sport a forte variance). Ce programme
donne une estimation raisonnee, pas une certitude.
==============================================================================
"""

import sys
import unicodedata
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import PoissonRegressor, LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Ce programme necessite scikit-learn : pip install scikit-learn")
    sys.exit(1)

import bracket_engine

DATA_DIR = "."
HOST_NATIONS = {"Mexico", "USA", "Canada"}


# ==========================================================================
# UTILITAIRES
# ==========================================================================
def normalize(text) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


# ==========================================================================
# CHARGEMENT & PREPARATION DES DONNEES
# ==========================================================================
class DataStore:
    def __init__(self, data_dir: str = DATA_DIR):
        d = data_dir.rstrip("/")
        self.data_dir = d

        self.teams = pd.read_csv(f"{d}/teams.csv").set_index("team_id")
        self.teams["is_host"] = self.teams["team_name"].isin(HOST_NATIONS).astype(int)

        self.venues = pd.read_csv(f"{d}/venues.csv").set_index("venue_id")
        self.referees = pd.read_csv(f"{d}/referees.csv").set_index("referee_id")
        self.stages = pd.read_csv(f"{d}/tournament_stages.csv").set_index("stage_id")

        raw_matches = pd.read_csv(f"{d}/matches.csv", parse_dates=["date"])
        self.mts = pd.read_csv(f"{d}/match_team_stats.csv")
        raw_events = pd.read_csv(f"{d}/match_events.csv")
        self.lineups = pd.read_csv(f"{d}/match_lineups.csv")

        self.squads = pd.read_csv(f"{d}/squads_and_players.csv")
        self.player_stats = pd.read_csv(f"{d}/player_stats.csv")

        self.features = pd.read_csv(f"{d}/match_prediction_features.csv",
                                     parse_dates=["date"])

        self.team_name_lookup = {
            normalize(row.team_name): tid for tid, row in self.teams.iterrows()
        }
        self._squad_agg = self._build_squad_aggregates()

        self.matches, self.bracket_log = bracket_engine.build_effective_matches(
            raw_matches, data_dir=d
        )
        self.events = self._merge_manual_events(raw_events)
        self.features = self._augment_features_with_manual_matches(self.features)

        self._long = self._build_long_format()
        self._add_rolling_form(self._long)

        self.poisson_model, self.poisson_scaler, self.poisson_features = \
            self._train_poisson_model()
        self.logit_model = self._train_logit_model()

    def _merge_manual_events(self, raw_events: pd.DataFrame) -> pd.DataFrame:
        extra = bracket_engine.build_effective_events(raw_events, data_dir=self.data_dir)
        if "_side" not in extra.columns:
            return extra
        new_mask = extra["_side"].notna()
        for idx in extra[new_mask].index:
            mid = extra.at[idx, "match_id"]
            side = extra.at[idx, "_side"]
            mrow = self.matches.loc[self.matches["match_id"] == mid]
            if len(mrow) == 0:
                continue
            mrow = mrow.iloc[0]
            extra.at[idx, "team_id"] = (mrow["home_team_id"] if side == "home"
                                         else mrow["away_team_id"])
        return extra.drop(columns=["_side"])

    def _augment_features_with_manual_matches(self, features: pd.DataFrame) -> pd.DataFrame:
        manual = bracket_engine.load_manual_results(self.data_dir)
        if len(manual) == 0:
            return features

        rows = []
        for _, mr in manual.iterrows():
            mid = int(mr["match_id"])
            mrow = self.matches.loc[self.matches["match_id"] == mid]
            if len(mrow) == 0:
                continue
            mrow = mrow.iloc[0]
            h, a = mrow["home_team_id"], mrow["away_team_id"]
            if pd.isna(h) or pd.isna(a):
                continue
            hs, as_ = mr["home_score"], mr["away_score"]
            result = "H" if hs > as_ else ("A" if as_ > hs else "D")
            val_h = self._squad_agg.loc[h, "avg_value"] if h in self._squad_agg.index else np.nan
            val_a = self._squad_agg.loc[a, "avg_value"] if a in self._squad_agg.index else np.nan
            rows.append({
                "match_id": mid, "date": mrow["date"],
                "home_team_id": h, "away_team_id": a,
                "home_elo": self.teams.loc[h, "elo_rating"],
                "away_elo": self.teams.loc[a, "elo_rating"],
                "home_fifa_rank": self.teams.loc[h, "fifa_ranking_pre_tournament"],
                "away_fifa_rank": self.teams.loc[a, "fifa_ranking_pre_tournament"],
                "home_squad_avg_value_eur": val_h,
                "away_squad_avg_value_eur": val_a,
                "home_is_host": self.teams.loc[h, "is_host"],
                "away_is_host": self.teams.loc[a, "is_host"],
                "match_result": result,
            })
        if not rows:
            return features
        return pd.concat([features, pd.DataFrame(rows)], ignore_index=True)

    # ---------------------------------------------------------------- #
    # Recherche tolerante
    # ---------------------------------------------------------------- #
    def find_team_id(self, query: str):
        key = normalize(query)
        if key in self.team_name_lookup:
            return self.team_name_lookup[key]
        matches = [(k, v) for k, v in self.team_name_lookup.items() if key in k]
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            return [self.teams.loc[v, "team_name"] for _, v in matches]
        return None

    def team_name(self, team_id) -> str:
        return self.teams.loc[team_id, "team_name"]

    def find_player(self, query: str):
        key = normalize(query)
        df = self.player_stats
        exact = df[df["player_name"].apply(normalize) == key]
        if len(exact) == 1:
            return exact.iloc[0]
        partial = df[df["player_name"].apply(lambda p: key in normalize(p))]
        if len(partial) == 1:
            return partial.iloc[0]
        if len(partial) > 1:
            return partial
        return None

    # ---------------------------------------------------------------- #
    # Preparation des donnees agregees
    # ---------------------------------------------------------------- #
    def _build_squad_aggregates(self) -> pd.DataFrame:
        sap = self.squads.copy()
        sap["birth_year"] = pd.to_datetime(sap["date_of_birth"], errors="coerce").dt.year
        agg = sap.groupby("team_id").agg(
            avg_value=("market_value_eur", "mean"),
            total_value=("market_value_eur", "sum"),
            avg_age=("birth_year", lambda x: 2026 - x.mean()),
            total_caps=("caps", "sum"),
        )
        return agg

    def _build_long_format(self) -> pd.DataFrame:
        """Une ligne par (match, equipe) -> permet de doubler l'echantillon
        d'entrainement (184 lignes au lieu de 92) pour le modele de buts."""
        completed = self.matches[self.matches["status"] == "Completed"].copy()

        home = completed[["match_id", "date", "home_team_id", "away_team_id",
                           "home_score", "away_score", "home_xg", "away_xg"]].copy()
        home.columns = ["match_id", "date", "team_id", "opp_id",
                        "goals_for", "goals_against", "xg_for", "xg_against"]
        home["is_home"] = 1

        away = completed[["match_id", "date", "away_team_id", "home_team_id",
                           "away_score", "home_score", "away_xg", "home_xg"]].copy()
        away.columns = ["match_id", "date", "team_id", "opp_id",
                        "goals_for", "goals_against", "xg_for", "xg_against"]
        away["is_home"] = 0

        long_df = pd.concat([home, away], ignore_index=True)
        long_df = long_df.merge(
            self.mts[["match_id", "team_id", "possession_pct",
                      "shots_on_target", "total_shots", "corners", "fouls"]],
            on=["match_id", "team_id"], how="left",
        )
        return long_df.sort_values(["team_id", "date"]).reset_index(drop=True)

    def _add_rolling_form(self, long_df: pd.DataFrame):
        """Moyennes cumulees SANS FUITE DE DONNEES (shift avant expanding) :
        pour le match N d'une equipe, on n'utilise que les matchs 1..N-1."""
        def prior_mean(col):
            return long_df.groupby("team_id")[col].transform(
                lambda s: s.shift().expanding().mean()
            )

        long_df["prev_goals_for"] = prior_mean("goals_for")
        long_df["prev_goals_against"] = prior_mean("goals_against")
        long_df["prev_xg_for"] = prior_mean("xg_for")
        long_df["prev_xg_against"] = prior_mean("xg_against")
        long_df["prev_sot"] = prior_mean("shots_on_target")

        # 1er match de chaque equipe : pas d'historique -> on comble avec
        # la moyenne generale de la competition (prior faible mais neutre)
        for col, src in [("prev_goals_for", "goals_for"),
                          ("prev_goals_against", "goals_against"),
                          ("prev_xg_for", "xg_for"),
                          ("prev_xg_against", "xg_against"),
                          ("prev_sot", "shots_on_target")]:
            long_df[col] = long_df[col].fillna(long_df[src].mean())

        long_df["elo"] = long_df["team_id"].map(self.teams["elo_rating"])
        long_df["opp_elo"] = long_df["opp_id"].map(self.teams["elo_rating"])
        long_df["rank"] = long_df["team_id"].map(self.teams["fifa_ranking_pre_tournament"])
        long_df["opp_rank"] = long_df["opp_id"].map(self.teams["fifa_ranking_pre_tournament"])
        long_df["is_host"] = long_df["team_id"].map(self.teams["is_host"])
        long_df["value"] = long_df["team_id"].map(self._squad_agg["avg_value"])
        long_df["opp_value"] = long_df["opp_id"].map(self._squad_agg["avg_value"])

        long_df["elo_diff"] = long_df["elo"] - long_df["opp_elo"]
        long_df["rank_diff"] = long_df["opp_rank"] - long_df["rank"]
        long_df["value_diff"] = np.log1p(long_df["value"]) - np.log1p(long_df["opp_value"])

    # ---------------------------------------------------------------- #
    # Entrainement des modeles
    # ---------------------------------------------------------------- #
    FEATURE_COLS = ["is_home", "elo_diff", "rank_diff", "value_diff", "is_host",
                    "prev_goals_for", "prev_goals_against",
                    "prev_xg_for", "prev_xg_against", "prev_sot"]

    def _train_poisson_model(self):
        return self.fit_poisson(self._long, self.FEATURE_COLS)

    def _train_logit_model(self):
        return self.fit_logit(self.features)

    # ---------------------------------------------------------------- #
    # v3 : methodes statiques reutilisees par prediction_tracker.py pour
    # re-entrainer les modeles "en aveugle du futur" (walk-forward), afin
    # de mesurer honnetement la fiabilite des predictions passees.
    # ---------------------------------------------------------------- #
    @staticmethod
    def fit_poisson(long_df: pd.DataFrame, feature_cols):
        X = long_df[feature_cols].values
        y = long_df["goals_for"].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        model = PoissonRegressor(alpha=1.0, max_iter=2000)
        model.fit(Xs, y)
        return model, scaler, feature_cols

    @staticmethod
    def fit_logit(features_df: pd.DataFrame):
        f = features_df.dropna(subset=["match_result"]).copy()
        f["elo_diff"] = f["home_elo"] - f["away_elo"]
        f["rank_diff"] = f["away_fifa_rank"] - f["home_fifa_rank"]
        f["value_diff"] = (np.log1p(f["home_squad_avg_value_eur"])
                            - np.log1p(f["away_squad_avg_value_eur"]))
        f["host_diff"] = f["home_is_host"] - f["away_is_host"]
        X = f[["elo_diff", "rank_diff", "value_diff", "host_diff"]].values
        y = f["match_result"].values
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X, y)
        return clf

    # ---------------------------------------------------------------- #
    # Etat "actuel" d'une equipe (pour predire un match futur/hypothetique)
    # ---------------------------------------------------------------- #
    def current_team_form(self, team_id) -> dict:
        rows = self._long[self._long["team_id"] == team_id]
        if len(rows) == 0:
            # equipe sans match joue -> moyenne generale de la competition
            return dict(
                goals_for=self._long["goals_for"].mean(),
                goals_against=self._long["goals_against"].mean(),
                xg_for=self._long["xg_for"].mean(),
                xg_against=self._long["xg_against"].mean(),
                sot=self._long["shots_on_target"].mean(),
            )
        return dict(
            goals_for=rows["goals_for"].mean(),
            goals_against=rows["goals_against"].mean(),
            xg_for=rows["xg_for"].mean(),
            xg_against=rows["xg_against"].mean(),
            sot=rows["shots_on_target"].mean(),
        )

    # ---------------------------------------------------------------- #
    # Stats "par jour"
    # ---------------------------------------------------------------- #
    def matches_on_date(self, date_str: str) -> pd.DataFrame:
        target = pd.to_datetime(date_str).normalize()
        return self.matches[self.matches["date"] == target].copy()

    # ---------------------------------------------------------------- #
    # Classement des groupes
    # ---------------------------------------------------------------- #
    def group_standings(self) -> dict:
        group_matches = self.matches[
            (self.matches["stage_id"] == 1) & (self.matches["status"] == "Completed")
        ]
        rows = {tid: dict(team=self.team_name(tid), P=0, W=0, D=0, L=0,
                           GF=0, GA=0, Pts=0)
                for tid in self.teams.index}

        for _, m in group_matches.iterrows():
            h, a = m["home_team_id"], m["away_team_id"]
            hs, as_ = m["home_score"], m["away_score"]
            rows[h]["P"] += 1
            rows[a]["P"] += 1
            rows[h]["GF"] += hs
            rows[h]["GA"] += as_
            rows[a]["GF"] += as_
            rows[a]["GA"] += hs
            if hs > as_:
                rows[h]["W"] += 1
                rows[h]["Pts"] += 3
                rows[a]["L"] += 1
            elif hs < as_:
                rows[a]["W"] += 1
                rows[a]["Pts"] += 3
                rows[h]["L"] += 1
            else:
                rows[h]["D"] += 1
                rows[a]["D"] += 1
                rows[h]["Pts"] += 1
                rows[a]["Pts"] += 1

        standings_df = pd.DataFrame(rows).T
        standings_df["GD"] = standings_df["GF"] - standings_df["GA"]
        standings_df["group"] = standings_df.index.map(self.teams["group_letter"])

        groups = {}
        for g, sub in standings_df.groupby("group"):
            sub = sub.sort_values(["Pts", "GD", "GF"], ascending=False)
            groups[g] = sub
        return groups


# ==========================================================================
# PREDICTION D'UN MATCH
# ==========================================================================
@dataclass
class MatchPrediction:
    team_a: str
    team_b: str
    lambda_a: float
    lambda_b: float
    score_matrix: dict
    proba_a: float
    proba_draw: float
    proba_b: float
    best_score: tuple
    logit_proba: dict  # {'H':.., 'D':.., 'A':..}


def predict_match(store: DataStore, team_a_id, team_b_id,
                   neutral: bool = False, max_goals: int = 7) -> MatchPrediction:
    teams = store.teams

    def build_row(team_id, opp_id, is_home_flag):
        form = store.current_team_form(team_id)
        opp_form = store.current_team_form(opp_id)
        elo_diff = teams.loc[team_id, "elo_rating"] - teams.loc[opp_id, "elo_rating"]
        rank_diff = teams.loc[opp_id, "fifa_ranking_pre_tournament"] - \
            teams.loc[team_id, "fifa_ranking_pre_tournament"]
        val_a = store._squad_agg.loc[team_id, "avg_value"] if team_id in store._squad_agg.index else 0
        val_b = store._squad_agg.loc[opp_id, "avg_value"] if opp_id in store._squad_agg.index else 0
        value_diff = np.log1p(val_a) - np.log1p(val_b)
        is_host = teams.loc[team_id, "is_host"]

        return [is_home_flag, elo_diff, rank_diff, value_diff, is_host,
                form["goals_for"], form["goals_against"],
                form["xg_for"], form["xg_against"], form["sot"]]

    home_flag_a = 0 if neutral else 1
    home_flag_b = 0 if neutral else 0

    row_a = build_row(team_a_id, team_b_id, home_flag_a)
    row_b = build_row(team_b_id, team_a_id, home_flag_b)

    Xs = store.poisson_scaler.transform(np.array([row_a, row_b]))
    lam_a, lam_b = store.poisson_model.predict(Xs)
    lam_a, lam_b = max(lam_a, 0.05), max(lam_b, 0.05)

    matrix = {}
    proba_a = proba_draw = proba_b = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_pmf(i, lam_a) * poisson_pmf(j, lam_b)
            matrix[(i, j)] = p
            if i > j:
                proba_a += p
            elif i == j:
                proba_draw += p
            else:
                proba_b += p
    best_score = max(matrix, key=matrix.get)

    # -- verification croisee par regression logistique --
    elo_diff = teams.loc[team_a_id, "elo_rating"] - teams.loc[team_b_id, "elo_rating"]
    rank_diff = teams.loc[team_b_id, "fifa_ranking_pre_tournament"] - \
        teams.loc[team_a_id, "fifa_ranking_pre_tournament"]
    val_a = store._squad_agg.loc[team_a_id, "avg_value"] if team_a_id in store._squad_agg.index else 0
    val_b = store._squad_agg.loc[team_b_id, "avg_value"] if team_b_id in store._squad_agg.index else 0
    value_diff = np.log1p(val_a) - np.log1p(val_b)
    host_diff = (0 if neutral else teams.loc[team_a_id, "is_host"]) - \
        (0 if neutral else teams.loc[team_b_id, "is_host"])

    logit_X = np.array([[elo_diff, rank_diff, value_diff, host_diff]])
    logit_proba = dict(zip(store.logit_model.classes_,
                            store.logit_model.predict_proba(logit_X)[0]))

    return MatchPrediction(
        team_a=store.team_name(team_a_id), team_b=store.team_name(team_b_id),
        lambda_a=lam_a, lambda_b=lam_b, score_matrix=matrix,
        proba_a=proba_a, proba_draw=proba_draw, proba_b=proba_b,
        best_score=best_score, logit_proba=logit_proba,
    )


# ==========================================================================
# AFFICHAGE
# ==========================================================================
def print_team_summary(store: DataStore, team_id):
    name = store.team_name(team_id)
    t = store.teams.loc[team_id]
    long_rows = store._long[store._long["team_id"] == team_id]

    print(f"\n{'='*64}")
    print(f" EQUIPE : {name}  ({t['fifa_code']})")
    print(f"{'='*64}")
    print(f"Confederation           : {t['confederation']}")
    print(f"Groupe                  : {t['group_letter']}")
    print(f"Classement FIFA pre-CM  : {t['fifa_ranking_pre_tournament']}")
    print(f"Elo rating              : {t['elo_rating']:.0f}")
    print(f"Selectionneur           : {t['manager_name']}")
    print(f"Nation hote             : {'Oui' if t['is_host'] else 'Non'}")

    if t.name in store._squad_agg.index:
        sq = store._squad_agg.loc[t.name]
        print(f"Valeur d'effectif totale: {sq['total_value']/1e6:.1f} M EUR "
              f"(moy. {sq['avg_value']/1e6:.1f} M EUR/joueur)")
        print(f"Age moyen de l'effectif : {sq['avg_age']:.1f} ans")

    if len(long_rows) == 0:
        print("\nAucun match joue pour l'instant.")
        return

    played = len(long_rows)
    wins = (long_rows["goals_for"] > long_rows["goals_against"]).sum()
    draws = (long_rows["goals_for"] == long_rows["goals_against"]).sum()
    losses = (long_rows["goals_for"] < long_rows["goals_against"]).sum()
    gf, ga = long_rows["goals_for"].sum(), long_rows["goals_against"].sum()

    print(f"\n-- Bilan ({played} matchs joues) --")
    print(f"V-N-D                   : {wins}-{draws}-{losses}")
    print(f"Buts marques/encaisses  : {gf:.0f} / {ga:.0f}  (diff. {gf-ga:+.0f})")
    print(f"xG marque/concede (moy) : {long_rows['xg_for'].mean():.2f} / "
          f"{long_rows['xg_against'].mean():.2f} par match")
    print(f"Possession moyenne      : {long_rows['possession_pct'].mean():.1f} %")
    print(f"Tirs cadres / match     : {long_rows['shots_on_target'].mean():.1f}")

    print("\n-- Historique des matchs --")
    for _, r in long_rows.sort_values("date").iterrows():
        opp = store.team_name(r["opp_id"])
        loc = "dom." if r["is_home"] else "ext."
        res = "V" if r["goals_for"] > r["goals_against"] else \
              ("N" if r["goals_for"] == r["goals_against"] else "D")
        print(f"  {r['date'].date()} [{loc}] vs {opp:<16} "
              f"{r['goals_for']:.0f}-{r['goals_against']:.0f} ({res})")

    print("\n-- Meilleurs joueurs (buts+passes) --")
    ps = store.player_stats[store.player_stats["team_id"] == team_id].copy()
    ps["ga"] = ps["goals"] + ps["assists"]
    top = ps.sort_values("ga", ascending=False).head(5)
    for _, p in top.iterrows():
        rating = f"note {p['average_rating']:.2f}" if pd.notna(p["average_rating"]) else ""
        print(f"  {p['player_name']:<26} {p['position']:<4} "
              f"buts={p['goals']:.0f} passes={p['assists']:.0f} {rating}")


def print_player(p: pd.Series):
    print(f"\n{'='*56}")
    print(f" JOUEUR : {p['player_name']}")
    print(f"{'='*56}")
    print(f"Poste                : {p['position']}")
    print(f"Matchs / titulaire   : {p['matches_played']:.0f} / {p['matches_started']:.0f}")
    print(f"Minutes jouees       : {p['minutes_played']:.0f}")
    print(f"Buts                 : {p['goals']:.0f}")
    print(f"Passes decisives     : {p['assists']:.0f}")
    if pd.notna(p.get("shots")):
        print(f"Tirs / cadres        : {p['shots']:.0f} / {p['shots_on_target']:.0f}")
    print(f"Cartons J / R        : {p['yellow_cards']:.0f} / {p['red_cards']:.0f}")
    if pd.notna(p.get("saves")):
        print(f"Arrets               : {p['saves']:.0f}")
        print(f"Buts encaisses       : {p['goals_conceded']:.0f}")
        print(f"Clean sheets         : {p['clean_sheets']:.0f}")
    if pd.notna(p.get("average_rating")):
        print(f"Note moyenne         : {p['average_rating']:.2f}")


def print_day_matches(store: DataStore, date_str: str):
    df = store.matches_on_date(date_str)
    if len(df) == 0:
        print(f"\nAucun match trouve le {date_str}.")
        return
    print(f"\n{'='*70}")
    print(f" MATCHS DU {date_str}")
    print(f"{'='*70}")
    for _, m in df.iterrows():
        stage = store.stages.loc[m["stage_id"], "stage_name"]
        venue = store.venues.loc[m["venue_id"], "stadium_name"] if pd.notna(m["venue_id"]) else "?"
        if m["status"] == "Completed":
            h = store.team_name(m["home_team_id"])
            a = store.team_name(m["away_team_id"])
            print(f"  [{m['match_id']:>3}] {stage:<16} {h:<15} {m['home_score']:.0f} - "
                  f"{m['away_score']:.0f} {a:<15} (xG {m['home_xg']:.2f}-{m['away_xg']:.2f})"
                  f"  @ {venue}")
        else:
            h = store.team_name(m["home_team_id"]) if pd.notna(m["home_team_id"]) else "?"
            a = store.team_name(m["away_team_id"]) if pd.notna(m["away_team_id"]) else "?"
            print(f"  [{m['match_id']:>3}] {stage:<16} {h:<15} vs {a:<15} "
                  f"-- {m['kickoff_time_utc']} UTC (a venir) @ {venue}")


def print_match_detail(store: DataStore, match_id: int):
    m = store.matches[store.matches["match_id"] == match_id]
    if len(m) == 0:
        print("Match introuvable.")
        return
    m = m.iloc[0]
    stage = store.stages.loc[m["stage_id"], "stage_name"]
    venue = store.venues.loc[m["venue_id"]]
    ref = store.referees.loc[m["referee_id"], "name"] if pd.notna(m["referee_id"]) else "?"
    h_name, a_name = store.team_name(m["home_team_id"]), store.team_name(m["away_team_id"])

    print(f"\n{'='*66}")
    print(f" MATCH #{match_id} - {stage} - {m['date'].date()}")
    print(f"{'='*66}")
    print(f"{h_name}  {m['home_score']:.0f} - {m['away_score']:.0f}  {a_name}"
          if m["status"] == "Completed" else f"{h_name} vs {a_name} (a venir)")
    print(f"Stade  : {venue['stadium_name']}, {venue['city']} ({venue['country']})")
    print(f"Arbitre: {ref}")

    if m["status"] != "Completed":
        return

    print(f"xG     : {m['home_xg']:.2f} - {m['away_xg']:.2f}")

    stats = store.mts[store.mts["match_id"] == match_id].set_index("team_id")
    if len(stats) == 2:
        h_s, a_s = stats.loc[m["home_team_id"]], stats.loc[m["away_team_id"]]
        print(f"\n{'Stat':<18}{h_name:>16}{a_name:>16}")
        for col, label in [("possession_pct", "Possession %"),
                           ("total_shots", "Tirs"),
                           ("shots_on_target", "Tirs cadres"),
                           ("corners", "Corners"),
                           ("fouls", "Fautes"),
                           ("offsides", "Hors-jeu"),
                           ("saves", "Arrets")]:
            print(f"{label:<18}{h_s[col]:>16.0f}{a_s[col]:>16.0f}")
        potm = h_s["player_of_the_match"] if pd.notna(h_s["player_of_the_match"]) else a_s["player_of_the_match"]
        if pd.notna(potm):
            print(f"\nHomme du match: {potm}")

    evs = store.events[store.events["match_id"] == match_id].sort_values("minute")
    if len(evs):
        print("\n-- Chronologie --")
        for _, e in evs.iterrows():
            team = store.team_name(e["team_id"])
            player_row = store.squads[store.squads["player_id"] == e["player_id"]]
            player = player_row.iloc[0]["player_name"] if len(player_row) else "?"
            print(f"  {e['minute']:>3}' {e['event_type']:<22} {player:<24} ({team})")


def print_group_standings(store: DataStore):
    groups = store.group_standings()
    print(f"\n{'='*66}")
    print(" CLASSEMENTS DE GROUPES (Phase de groupes)")
    print(f"{'='*66}")
    for g in sorted(groups):
        print(f"\nGroupe {g}")
        print(f"{'Equipe':<16}{'P':>3}{'V':>3}{'N':>3}{'D':>3}{'BM':>4}{'BE':>4}{'Diff':>6}{'Pts':>5}")
        for tid, row in groups[g].iterrows():
            print(f"{row['team']:<16}{row['P']:>3.0f}{row['W']:>3.0f}{row['D']:>3.0f}"
                  f"{row['L']:>3.0f}{row['GF']:>4.0f}{row['GA']:>4.0f}{row['GD']:>+6.0f}{row['Pts']:>5.0f}")


def print_prediction(pred: MatchPrediction):
    print(f"\n{'='*62}")
    print(f" PREDICTION : {pred.team_a}  vs  {pred.team_b}")
    print(f"{'='*62}")
    print(f"[Modele Poisson] Buts attendus : {pred.team_a} = {pred.lambda_a:.2f}  "
          f"| {pred.team_b} = {pred.lambda_b:.2f}")
    print(f"Score le plus probable          : {pred.team_a} {pred.best_score[0]} - "
          f"{pred.best_score[1]} {pred.team_b}  "
          f"({pred.score_matrix[pred.best_score]*100:.1f}%)")

    print("\nProbabilites d'issue (modele de Poisson - simule tous les scores) :")
    print(f"  Victoire {pred.team_a:<15}: {fmt_pct(pred.proba_a)}")
    print(f"  Match nul               : {fmt_pct(pred.proba_draw)}")
    print(f"  Victoire {pred.team_b:<15}: {fmt_pct(pred.proba_b)}")

    print("\nProbabilites d'issue (verification croisee - regression logistique) :")
    lp = pred.logit_proba
    print(f"  Victoire {pred.team_a:<15}: {fmt_pct(lp.get('H', 0))}")
    print(f"  Match nul               : {fmt_pct(lp.get('D', 0))}")
    print(f"  Victoire {pred.team_b:<15}: {fmt_pct(lp.get('A', 0))}")

    print("\nTop 5 scores les plus probables (modele de Poisson) :")
    top_scores = sorted(pred.score_matrix.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for (i, j), p in top_scores:
        print(f"  {pred.team_a} {i} - {j} {pred.team_b}   ({p*100:.1f}%)")


# ==========================================================================
# MENU INTERACTIF
# ==========================================================================
def choose_team(store: DataStore, prompt: str):
    while True:
        query = input(prompt).strip()
        if query == "":
            return None
        result = store.find_team_id(query)
        if result is None:
            print("  -> Aucune equipe trouvee.")
        elif isinstance(result, list):
            print(f"  -> Plusieurs equipes correspondent : {', '.join(result)}")
        else:
            return result


def menu():
    print("Chargement des donnees et entrainement des modeles...")
    store = DataStore(DATA_DIR)
    print(f"{len(store.teams)} equipes, "
          f"{(store.matches['status']=='Completed').sum()} matchs joues, "
          f"{len(store.player_stats)} joueurs suivis.\n")

    while True:
        print(f"\n{'#'*62}")
        print("# FIFA WORLD CUP 2026 - ANALYTICS & PREDICTION")
        print(f"{'#'*62}")
        print("1. Classement des groupes")
        print("2. Statistiques d'une equipe")
        print("3. Statistiques d'un joueur")
        print("4. Matchs d'un jour (stats du jour)")
        print("5. Detail complet d'un match (score, xG, cartons, compositions...)")
        print("6. Predire un match entre 2 equipes")
        print("0. Quitter")
        choice = input("\nVotre choix : ").strip()

        if choice == "1":
            print_group_standings(store)

        elif choice == "2":
            tid = choose_team(store, "Nom de l'equipe : ")
            if tid:
                print_team_summary(store, tid)

        elif choice == "3":
            query = input("Nom du joueur : ").strip()
            result = store.find_player(query)
            if result is None:
                print("  -> Joueur introuvable.")
            elif isinstance(result, pd.DataFrame):
                print("  -> Plusieurs joueurs correspondent :")
                for _, p in result.iterrows():
                    print(f"     - {p['player_name']} ({store.team_name(p['team_id'])})")
            else:
                print_player(result)

        elif choice == "4":
            date_str = input("Date (AAAA-MM-JJ), ex 2026-06-11 : ").strip()
            print_day_matches(store, date_str)

        elif choice == "5":
            try:
                mid = int(input("Numero du match (match_id) : ").strip())
                print_match_detail(store, mid)
            except ValueError:
                print("  -> Identifiant invalide.")

        elif choice == "6":
            team_a = choose_team(store, "Equipe A : ")
            if not team_a:
                continue
            team_b = choose_team(store, "Equipe B : ")
            if not team_b:
                continue
            if team_a == team_b:
                print("  -> Choisissez deux equipes differentes.")
                continue
            neutral_in = input("Terrain neutre ? (o/N) : ").strip().lower()
            neutral = neutral_in == "o"
            pred = predict_match(store, team_a, team_b, neutral=neutral)
            print_prediction(pred)

        elif choice == "0":
            print("A bientot !")
            sys.exit(0)

        else:
            print("Choix invalide.")


if __name__ == "__main__":
    menu()
