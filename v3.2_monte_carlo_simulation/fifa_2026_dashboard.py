#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 FIFA WORLD CUP 2026 - DASHBOARD GRAPHIQUE (Streamlit)
==============================================================================
Interface graphique par-dessus le moteur d'analyse/prediction de
fifa_2026_app.py (meme dossier requis). Lance avec :

    streamlit run fifa_2026_dashboard.py

Installation prealable si besoin :
    pip install streamlit plotly pandas numpy scikit-learn
==============================================================================
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fifa_2026_app import DataStore, predict_match, poisson_pmf
import bracket_engine
import prediction_tracker as pt
import odds_manager as om
import tournament_simulator as ts

st.set_page_config(page_title="FIFA World Cup 2026 - Analytics", layout="wide",
                    page_icon="⚽")


# ==========================================================================
# CHARGEMENT (mis en cache : les modeles ne sont entraines qu'une fois...
# ... et re-entraines automatiquement des qu'un resultat est saisi a la
# main, grace au parametre de version base sur la date de derniere
# modification de manual_results.csv, qui invalide le cache Streamlit.)
# ==========================================================================
def _manual_results_version() -> float:
    path = os.path.join(".", "manual_results.csv")
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_resource(show_spinner="Chargement des donnees et entrainement des modeles...")
def load_store(version: float):
    return DataStore(".")


try:
    store = load_store(_manual_results_version())
except FileNotFoundError as e:
    st.error(
        f"Fichier introuvable : {e}\n\n"
        "Verifie que fifa_2026_dashboard.py est bien dans le MEME DOSSIER "
        "que tous les fichiers .csv et fifa_2026_app.py."
    )
    st.stop()

TEAM_NAMES = sorted(store.teams["team_name"].tolist())


def team_id_of(name: str):
    return store.find_team_id(name)


# ==========================================================================
# SIDEBAR - NAVIGATION
# ==========================================================================
st.sidebar.title("⚽ World Cup 2026")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Vue d'ensemble", "📊 Classements de groupes", "🏳️ Statistiques equipe",
     "🧑 Statistiques joueur", "📅 Matchs d'un jour", "🔍 Detail d'un match",
     "🔮 Prediction", "✏️ Saisie manuelle", "🎯 Fiabilite de l'IA",
     "🎲 Paris sportifs", "🌐 Cotes reelles (OddsPortal)",
     "📐 Calibration des modeles", "🌀 Simulation Monte Carlo"],
)

n_pending = len(bracket_engine.pending_playable_matches(store.matches, data_dir="."))
if n_pending:
    st.sidebar.warning(f"{n_pending} match(s) en attente de saisie manuelle.")

n_played = int((store.matches["status"] == "Completed").sum())
n_total = len(store.matches)
st.sidebar.markdown("---")
st.sidebar.metric("Matchs joues", f"{n_played} / {n_total}")
st.sidebar.metric("Equipes", len(store.teams))
st.sidebar.metric("Joueurs suivis", len(store.player_stats))
st.sidebar.caption(
    "Modele de prediction : regression de Poisson (buts attendus) "
    "+ regression logistique (verification croisee), entrainees sur "
    "les matchs deja joues."
)


# ==========================================================================
# PAGE : VUE D'ENSEMBLE
# ==========================================================================
if page == "🏠 Vue d'ensemble":
    st.title("Vue d'ensemble de la competition")

    completed = store.matches[store.matches["status"] == "Completed"]
    total_goals = completed["home_score"].sum() + completed["away_score"].sum()
    avg_goals = total_goals / len(completed) if len(completed) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matchs joues", n_played)
    c2.metric("Buts marques", f"{total_goals:.0f}")
    c3.metric("Buts / match", f"{avg_goals:.2f}")
    c4.metric("Matchs a venir", n_total - n_played)

    st.subheader("Meilleurs buteurs du tournoi")
    top_scorers = store.player_stats.sort_values("goals", ascending=False).head(10).copy()
    top_scorers["team"] = top_scorers["team_id"].map(store.teams["team_name"])
    fig = px.bar(top_scorers, x="goals", y="player_name", color="team",
                 orientation="h", labels={"goals": "Buts", "player_name": "Joueur"},
                 title="Top 10 buteurs")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Meilleures attaques (buts/match)")
        gf = store._long.groupby("team_id")["goals_for"].mean().sort_values(ascending=False).head(10)
        gf.index = gf.index.map(store.teams["team_name"])
        fig = px.bar(gf, orientation="h", labels={"value": "Buts/match", "index": "Equipe"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Meilleures defenses (buts encaisses/match)")
        ga = store._long.groupby("team_id")["goals_against"].mean().sort_values().head(10)
        ga.index = ga.index.map(store.teams["team_name"])
        fig = px.bar(ga, orientation="h", labels={"value": "Buts encaisses/match", "index": "Equipe"})
        fig.update_layout(yaxis={"categoryorder": "total descending"}, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE : CLASSEMENTS DE GROUPES
# ==========================================================================
elif page == "📊 Classements de groupes":
    st.title("Classements de la phase de groupes")
    groups = store.group_standings()

    cols = st.columns(3)
    for i, g in enumerate(sorted(groups)):
        df = groups[g][["team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]].reset_index(drop=True)
        df.columns = ["Equipe", "J", "V", "N", "D", "BM", "BE", "Diff", "Pts"]
        with cols[i % 3]:
            st.markdown(f"**Groupe {g}**")
            st.dataframe(df, hide_index=True, use_container_width=True)


# ==========================================================================
# PAGE : STATISTIQUES EQUIPE
# ==========================================================================
elif page == "🏳️ Statistiques equipe":
    st.title("Statistiques d'une equipe")
    team_name = st.selectbox("Choisir une equipe", TEAM_NAMES)
    tid = team_id_of(team_name)
    t = store.teams.loc[tid]
    rows = store._long[store._long["team_id"] == tid].sort_values("date")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Groupe", t["group_letter"])
    c2.metric("Classement FIFA", int(t["fifa_ranking_pre_tournament"]))
    c3.metric("Elo", int(t["elo_rating"]))
    c4.metric("Nation hote", "Oui" if t["is_host"] else "Non")

    if tid in store._squad_agg.index:
        sq = store._squad_agg.loc[tid]
        c1, c2, c3 = st.columns(3)
        c1.metric("Valeur d'effectif", f"{sq['total_value']/1e6:.0f} M€")
        c2.metric("Age moyen", f"{sq['avg_age']:.1f} ans")
        c3.metric("Selectionneur", t["manager_name"])

    if len(rows) == 0:
        st.info("Cette equipe n'a pas encore joue de match.")
    else:
        wins = (rows["goals_for"] > rows["goals_against"]).sum()
        draws = (rows["goals_for"] == rows["goals_against"]).sum()
        losses = (rows["goals_for"] < rows["goals_against"]).sum()

        st.subheader(f"Bilan ({len(rows)} matchs) : {wins}V - {draws}N - {losses}D")

        c1, c2 = st.columns(2)
        with c1:
            plot_df = rows.copy()
            plot_df["adversaire"] = plot_df["opp_id"].map(store.teams["team_name"])
            plot_df["match"] = plot_df["date"].dt.strftime("%d/%m") + " vs " + plot_df["adversaire"]
            fig = go.Figure()
            fig.add_bar(x=plot_df["match"], y=plot_df["goals_for"], name="Buts marques")
            fig.add_bar(x=plot_df["match"], y=plot_df["goals_against"], name="Buts encaisses")
            fig.update_layout(title="Buts marques / encaisses par match", barmode="group", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=plot_df["match"], y=plot_df["xg_for"], name="xG pour", mode="lines+markers")
            fig.add_scatter(x=plot_df["match"], y=plot_df["xg_against"], name="xG contre", mode="lines+markers")
            fig.update_layout(title="Evolution du xG", height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Meilleurs joueurs (buts + passes)")
        ps = store.player_stats[store.player_stats["team_id"] == tid].copy()
        ps["ga"] = ps["goals"] + ps["assists"]
        top = ps.sort_values("ga", ascending=False).head(8)
        fig = px.bar(top, x="ga", y="player_name", orientation="h",
                     hover_data=["goals", "assists"],
                     labels={"ga": "Buts + passes", "player_name": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Historique des matchs")
        hist = rows.copy()
        hist["Adversaire"] = hist["opp_id"].map(store.teams["team_name"])
        hist["Lieu"] = hist["is_home"].map({1: "Domicile", 0: "Exterieur"})
        hist["Score"] = hist["goals_for"].astype(int).astype(str) + "-" + hist["goals_against"].astype(int).astype(str)
        hist["Date"] = hist["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(hist[["Date", "Lieu", "Adversaire", "Score", "xg_for", "xg_against",
                            "possession_pct"]].rename(columns={
            "xg_for": "xG pour", "xg_against": "xG contre", "possession_pct": "Possession %"
        }), hide_index=True, use_container_width=True)


# ==========================================================================
# PAGE : STATISTIQUES JOUEUR
# ==========================================================================
elif page == "🧑 Statistiques joueur":
    st.title("Statistiques d'un joueur")

    filter_team = st.selectbox("Filtrer par equipe (optionnel)", ["Toutes"] + TEAM_NAMES)
    ps = store.player_stats.copy()
    ps["team"] = ps["team_id"].map(store.teams["team_name"])
    if filter_team != "Toutes":
        ps = ps[ps["team"] == filter_team]

    player_name = st.selectbox("Choisir un joueur", sorted(ps["player_name"].tolist()))
    p = ps[ps["player_name"] == player_name].iloc[0]

    st.subheader(f"{p['player_name']} — {p['team']} ({p['position']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matchs joues", int(p["matches_played"]))
    c2.metric("Buts", int(p["goals"]))
    c3.metric("Passes decisives", int(p["assists"]))
    c4.metric("Minutes jouees", int(p["minutes_played"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Cartons jaunes", int(p["yellow_cards"]))
    c2.metric("Cartons rouges", int(p["red_cards"]))
    if pd.notna(p.get("average_rating")):
        c3.metric("Note moyenne", f"{p['average_rating']:.2f}")

    if p["position"] == "GK" and pd.notna(p.get("saves")):
        st.subheader("Statistiques de gardien")
        c1, c2, c3 = st.columns(3)
        c1.metric("Arrets", int(p["saves"]))
        c2.metric("Buts encaisses", int(p["goals_conceded"]))
        c3.metric("Clean sheets", int(p["clean_sheets"]))


# ==========================================================================
# PAGE : MATCHS D'UN JOUR
# ==========================================================================
elif page == "📅 Matchs d'un jour":
    st.title("Matchs d'un jour donne")

    min_d, max_d = store.matches["date"].min().date(), store.matches["date"].max().date()
    chosen_date = st.date_input("Choisir une date", value=min_d, min_value=min_d, max_value=max_d)

    day_matches = store.matches[store.matches["date"] == pd.to_datetime(chosen_date)]

    if len(day_matches) == 0:
        st.info("Aucun match ce jour-la.")
    else:
        for _, m in day_matches.iterrows():
            stage = store.stages.loc[m["stage_id"], "stage_name"]
            venue = store.venues.loc[m["venue_id"], "stadium_name"] if pd.notna(m["venue_id"]) else "?"
            h = store.team_name(m["home_team_id"]) if pd.notna(m["home_team_id"]) else "?"
            a = store.team_name(m["away_team_id"]) if pd.notna(m["away_team_id"]) else "?"

            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 3])
                if m["status"] == "Completed":
                    c1.markdown(f"### {h}")
                    c2.markdown(f"### {m['home_score']:.0f} - {m['away_score']:.0f}")
                    c3.markdown(f"### {a}")
                    st.caption(f"{stage} · xG {m['home_xg']:.2f} - {m['away_xg']:.2f} · {venue} "
                               f"· match #{m['match_id']}")
                else:
                    c1.markdown(f"### {h}")
                    c2.markdown("### vs")
                    c3.markdown(f"### {a}")
                    st.caption(f"{stage} · {m['kickoff_time_utc']} UTC (a venir) · {venue} "
                               f"· match #{m['match_id']}")


# ==========================================================================
# PAGE : DETAIL D'UN MATCH
# ==========================================================================
elif page == "🔍 Detail d'un match":
    st.title("Detail complet d'un match")

    m_display = store.matches.copy()
    m_display["home_name"] = m_display["home_team_id"].map(store.teams["team_name"])
    m_display["away_name"] = m_display["away_team_id"].map(store.teams["team_name"])
    m_display["label"] = (m_display["date"].dt.strftime("%Y-%m-%d") + " — " +
                           m_display["home_name"].fillna("?") + " vs " +
                           m_display["away_name"].fillna("?") +
                           " (#" + m_display["match_id"].astype(str) + ")")

    label = st.selectbox("Choisir un match", m_display.sort_values("date")["label"].tolist())
    match_id = int(label.split("#")[-1].rstrip(")"))
    m = store.matches[store.matches["match_id"] == match_id].iloc[0]

    stage = store.stages.loc[m["stage_id"], "stage_name"]
    venue = store.venues.loc[m["venue_id"]]
    ref = store.referees.loc[m["referee_id"], "name"] if pd.notna(m["referee_id"]) else "?"
    h_name, a_name = store.team_name(m["home_team_id"]), store.team_name(m["away_team_id"])

    st.subheader(f"{stage} — {m['date'].date()}")
    c1, c2, c3 = st.columns([2, 1, 2])
    c1.markdown(f"## {h_name}")
    if m["status"] == "Completed":
        c2.markdown(f"## {m['home_score']:.0f} - {m['away_score']:.0f}")
    else:
        c2.markdown("## vs")
    c3.markdown(f"## {a_name}")
    st.caption(f"🏟️ {venue['stadium_name']}, {venue['city']} ({venue['country']}) · 👤 Arbitre : {ref}")

    if m["status"] != "Completed":
        st.info("Ce match n'a pas encore ete joue.")
    else:
        stats = store.mts[store.mts["match_id"] == match_id].set_index("team_id")
        if len(stats) == 2:
            h_s, a_s = stats.loc[m["home_team_id"]], stats.loc[m["away_team_id"]]

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=[m["home_xg"], m["away_xg"]], y=[h_name, a_name],
                    orientation="h", marker_color=["#1f77b4", "#ff7f0e"]))
                fig.update_layout(title="xG (buts attendus)", height=250)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                if pd.notna(h_s["player_of_the_match"]) or pd.notna(a_s["player_of_the_match"]):
                    potm = h_s["player_of_the_match"] if pd.notna(h_s["player_of_the_match"]) else a_s["player_of_the_match"]
                    st.metric("🏅 Homme du match", potm)

            comp_stats = ["possession_pct", "total_shots", "shots_on_target", "corners", "fouls", "offsides", "saves"]
            labels = ["Possession %", "Tirs", "Tirs cadres", "Corners", "Fautes", "Hors-jeu", "Arrets"]
            fig = go.Figure()
            fig.add_bar(y=labels, x=[h_s[c] for c in comp_stats], name=h_name, orientation="h")
            fig.add_bar(y=labels, x=[-a_s[c] for c in comp_stats], name=a_name, orientation="h")
            fig.update_layout(title="Comparaison des statistiques", barmode="relative",
                               xaxis_title="", height=400)
            st.plotly_chart(fig, use_container_width=True)

        evs = store.events[store.events["match_id"] == match_id].sort_values("minute").copy()
        if len(evs):
            st.subheader("Chronologie du match")
            evs["Equipe"] = evs["team_id"].map(store.teams["team_name"])
            evs["Joueur"] = evs["player_id"].map(
                store.squads.set_index("player_id")["player_name"]
            )
            st.dataframe(
                evs[["minute", "event_type", "Joueur", "Equipe"]].rename(
                    columns={"minute": "Minute", "event_type": "Evenement"}),
                hide_index=True, use_container_width=True,
            )


# ==========================================================================
# PAGE : PREDICTION
# ==========================================================================
elif page == "🔮 Prediction":
    st.title("Predire un match")
    st.caption(
        "Modele de Poisson (buts attendus) entraine sur les matchs deja joues, "
        "avec verification croisee par regression logistique."
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    team_a = c1.selectbox("Equipe A (domicile)", TEAM_NAMES, index=TEAM_NAMES.index("France") if "France" in TEAM_NAMES else 0)
    team_b = c2.selectbox("Equipe B (exterieur)", TEAM_NAMES, index=TEAM_NAMES.index("Brazil") if "Brazil" in TEAM_NAMES else 1)
    neutral = c3.checkbox("Terrain neutre", value=True)

    if team_a == team_b:
        st.warning("Choisis deux equipes differentes.")
    else:
        tid_a, tid_b = team_id_of(team_a), team_id_of(team_b)
        pred = predict_match(store, tid_a, tid_b, neutral=neutral)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"⚽ Buts attendus — {pred.team_a}", f"{pred.lambda_a:.2f}")
        c2.metric("Score le plus probable",
                  f"{pred.best_score[0]} - {pred.best_score[1]}",
                  f"{pred.score_matrix[pred.best_score]*100:.1f}% de proba")
        c3.metric(f"⚽ Buts attendus — {pred.team_b}", f"{pred.lambda_b:.2f}")

        st.subheader("Probabilites d'issue")
        cmp_df = pd.DataFrame({
            "Issue": [f"Victoire {pred.team_a}", "Match nul", f"Victoire {pred.team_b}"],
            "Modele Poisson": [pred.proba_a, pred.proba_draw, pred.proba_b],
            "Regression logistique": [pred.logit_proba.get("H", 0),
                                       pred.logit_proba.get("D", 0),
                                       pred.logit_proba.get("A", 0)],
        })
        fig = px.bar(cmp_df.melt(id_vars="Issue", var_name="Modele", value_name="Probabilite"),
                     x="Issue", y="Probabilite", color="Modele", barmode="group",
                     text_auto=".0%")
        fig.update_layout(yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Matrice des scores probables (loi de Poisson)")
        max_g = 6
        z = np.array([[pred.score_matrix.get((i, j), 0) for j in range(max_g + 1)]
                       for i in range(max_g + 1)])
        fig = go.Figure(data=go.Heatmap(
            z=z * 100, x=[str(j) for j in range(max_g + 1)], y=[str(i) for i in range(max_g + 1)],
            colorscale="Blues", texttemplate="%{z:.1f}%", hoverinfo="skip"))
        fig.update_layout(
            title=f"Probabilite (%) de chaque score exact — lignes = buts {pred.team_a}, colonnes = buts {pred.team_b}",
            xaxis_title=pred.team_b, yaxis_title=pred.team_a, height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 5 des scores les plus probables")
        top5 = sorted(pred.score_matrix.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top5_df = pd.DataFrame([
            {"Score": f"{pred.team_a} {i} - {j} {pred.team_b}", "Probabilite": f"{p*100:.1f}%"}
            for (i, j), p in top5
        ])
        st.dataframe(top5_df, hide_index=True, use_container_width=True)


# ==========================================================================
# PAGE : SAISIE MANUELLE (v3)
# ==========================================================================
elif page == "✏️ Saisie manuelle":
    st.title("Saisie manuelle des resultats")
    st.caption(
        "Un match apparait ici automatiquement une fois que ses 2 equipes "
        "sont connues (le bracket se met a jour tout seul quand un tour "
        "precedent est saisi). Score + buteurs uniquement. "
        "⚠️ La saisie est DEFINITIVE : impossible de la corriger ensuite."
    )

    pending = bracket_engine.pending_playable_matches(store.matches, data_dir=".")
    if len(pending) == 0:
        st.info("Aucun match en attente de saisie pour le moment — "
                 "soit tout est deja joue, soit on attend encore le "
                 "resultat d'un tour precedent.")
    else:
        pending = pending.copy()
        pending["home_name"] = pending["home_team_id"].map(store.teams["team_name"])
        pending["away_name"] = pending["away_team_id"].map(store.teams["team_name"])
        pending["stage_name"] = pending["stage_id"].map(store.stages["stage_name"])
        pending["label"] = (
            pending["stage_name"] + " — " + pending["home_name"] + " vs "
            + pending["away_name"] + " (#" + pending["match_id"].astype(str) + ")"
        )

        label = st.selectbox("Choisir le match a saisir", pending["label"].tolist())
        mrow = pending[pending["label"] == label].iloc[0]
        match_id = int(mrow["match_id"])
        home_id, away_id = int(mrow["home_team_id"]), int(mrow["away_team_id"])
        home_name, away_name = mrow["home_name"], mrow["away_name"]
        is_knockout = bool(store.stages.loc[mrow["stage_id"], "is_knockout"])

        st.subheader(f"{home_name}  vs  {away_name}")

        c1, c2 = st.columns(2)
        home_score = c1.number_input(f"Buts — {home_name}", min_value=0, max_value=15,
                                      value=0, step=1, key=f"hs_{match_id}")
        away_score = c2.number_input(f"Buts — {away_name}", min_value=0, max_value=15,
                                      value=0, step=1, key=f"as_{match_id}")

        home_pen = away_pen = None
        penalty_conflict = False
        if is_knockout and home_score == away_score:
            st.warning("Match nul en phase a elimination directe → tirs au but obligatoires.")
            c1, c2 = st.columns(2)
            home_pen = c1.number_input(f"Tirs au but — {home_name}", min_value=0,
                                        max_value=25, value=0, step=1, key=f"hp_{match_id}")
            away_pen = c2.number_input(f"Tirs au but — {away_name}", min_value=0,
                                        max_value=25, value=0, step=1, key=f"ap_{match_id}")
            penalty_conflict = (home_pen == away_pen)

        home_players = store.squads[store.squads["team_id"] == home_id].sort_values("player_name")
        away_players = store.squads[store.squads["team_id"] == away_id].sort_values("player_name")
        home_options = dict(zip(home_players["player_name"], home_players["player_id"]))
        away_options = dict(zip(away_players["player_name"], away_players["player_id"]))

        home_scorers = []
        if home_score > 0:
            st.markdown(f"**Buteurs — {home_name}** *(un joueur peut etre choisi plusieurs fois pour un doublé/triplé)*")
            cols = st.columns(min(int(home_score), 4))
            for i in range(int(home_score)):
                name = cols[i % len(cols)].selectbox(
                    f"But n°{i + 1}", list(home_options.keys()), key=f"hsc_{match_id}_{i}")
                home_scorers.append(int(home_options[name]))

        away_scorers = []
        if away_score > 0:
            st.markdown(f"**Buteurs — {away_name}**")
            cols = st.columns(min(int(away_score), 4))
            for i in range(int(away_score)):
                name = cols[i % len(cols)].selectbox(
                    f"But n°{i + 1}", list(away_options.keys()), key=f"asc_{match_id}_{i}")
                away_scorers.append(int(away_options[name]))

        st.markdown("---")
        if penalty_conflict:
            st.error("Les tirs au but ne peuvent pas etre a egalite entre les 2 equipes.")
        else:
            confirm = st.checkbox(
                "Je confirme ce resultat — je sais qu'il ne pourra plus etre modifie.",
                key=f"confirm_{match_id}",
            )
            if st.button("✅ Valider ce resultat", type="primary", disabled=not confirm):
                ok, msg = bracket_engine.save_manual_result(
                    match_id, int(home_score), int(away_score),
                    home_scorers=home_scorers, away_scorers=away_scorers,
                    home_penalty_score=home_pen, away_penalty_score=away_pen,
                    data_dir=".",
                )
                if ok:
                    st.success(msg + " Le bracket et les modeles IA viennent d'etre mis a jour.")
                    st.rerun()
                else:
                    st.error(msg)


# ==========================================================================
# PAGE : FIABILITE DE L'IA (v3)
# ==========================================================================
elif page == "🎯 Fiabilite de l'IA":
    st.title("Fiabilite des predictions de l'IA face au reel")
    st.caption(
        "Pour chaque match deja joue, on reconstitue la prediction que "
        "l'IA aurait faite en ne connaissant QUE les matchs anterieurs a "
        "sa date (le modele est ré-entraine specifiquement pour chaque "
        "match evalue, aucune fuite du futur). Deux mesures distinctes : "
        "le resultat (Victoire/Nul/Defaite) et le score exact."
    )

    with st.spinner("Calcul des predictions retrospectives (walk-forward)..."):
        evals = pt.evaluate_predictions(store)

    if len(evals) == 0:
        st.info("Pas encore assez de matchs joues pour evaluer les predictions.")
    else:
        summ = pt.summarize(evals)

        st.subheader("1) Resultat — Victoire / Nul / Defaite")
        c1, c2 = st.columns(2)
        c1.metric("Bonne issue, choix n°1 de l'IA", f"{summ['outcome_top1_pct']*100:.1f}%")
        c2.metric("Issue reelle dans le top 2 de l'IA", f"{summ['outcome_top2_pct']*100:.1f}%")

        st.subheader("2) Score exact")
        c1, c2, c3 = st.columns(3)
        c1.metric("Score exact = choix n°1", f"{summ['score_top1_pct']*100:.1f}%")
        c2.metric("Score exact dans le top 2", f"{summ['score_top2_pct']*100:.1f}%")
        c3.metric("Score exact dans le top 3", f"{summ['score_top3_pct']*100:.1f}%")

        st.caption(
            f"Base sur {summ['n_matches_evaluated']} matchs evalues. "
            "Les tout premiers matchs du tournoi sont exclus par manque "
            "d'historique pour entrainer un modele representatif de "
            "l'epoque."
        )

        st.subheader("Precision (résultat) par phase de la competition")
        stage_df = summ["accuracy_by_stage"].reset_index()
        stage_df.columns = ["Phase", "Precision"]
        fig = px.bar(stage_df, x="Phase", y="Precision", text_auto=".0%")
        fig.update_layout(yaxis_tickformat=".0%", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Plus grosses surprises")
        st.caption("Matchs ou l'issue reelle avait la plus faible probabilite selon l'IA.")
        upsets = summ["biggest_upsets"][
            ["date", "home_team", "away_team", "actual_score",
             "predicted_best_score", "outcome_proba_of_actual"]
        ].copy()
        upsets["date"] = pd.to_datetime(upsets["date"]).dt.strftime("%Y-%m-%d")
        upsets["outcome_proba_of_actual"] = (
            (upsets["outcome_proba_of_actual"] * 100).round(1).astype(str) + " %"
        )
        upsets.columns = ["Date", "Domicile", "Exterieur", "Score reel",
                           "Score predit (top 1)", "Probabilite de l'issue reelle"]
        st.dataframe(upsets, hide_index=True, use_container_width=True)

        st.subheader("Detail match par match")
        detail = evals[[
            "date", "home_team", "away_team", "actual_score", "predicted_best_score",
            "score_rank", "actual_outcome", "predicted_best_outcome", "outcome_rank", "source",
        ]].copy()
        detail["date"] = pd.to_datetime(detail["date"]).dt.strftime("%Y-%m-%d")
        detail["source"] = detail["source"].map({"dataset": "Dataset", "manual": "Saisie manuelle"})
        detail.columns = ["Date", "Domicile", "Exterieur", "Score reel", "Score predit (top 1)",
                           "Rang du score reel", "Issue reelle", "Issue predite (top 1)",
                           "Rang de l'issue reelle", "Source"]
        st.dataframe(detail.sort_values("Date", ascending=False),
                     hide_index=True, use_container_width=True)


# ==========================================================================
# PAGE : PARIS SPORTIFS
# ==========================================================================
elif page == "🎲 Paris sportifs":
    st.title("Simulation de paris sportifs — 1N2 & double chance")
    st.caption(
        "⚠️ Le dataset ne contient aucune cote de bookmaker. Les cotes "
        "ci-dessous sont des **cotes equitables** (implied odds, sans marge) "
        "calculees a partir des probabilites retrospectives (walk-forward) "
        "des 2 modeles de l'IA : cote = 1 / probabilite. Elles servent a "
        "comparer les 2 modeles entre eux, pas a representer un vrai "
        "marche de paris."
    )

    stake = st.number_input(
        "Mise par match simulee (EUR)", min_value=1.0, max_value=1000.0,
        value=pt.STAKE_DEFAULT, step=5.0,
    )

    with st.spinner("Calcul des predictions retrospectives et des cotes (walk-forward)..."):
        evals = pt.evaluate_predictions(store)

    if len(evals) == 0:
        st.info("Pas encore assez de matchs joues pour simuler des paris.")
    else:
        bets = pt.summarize_betting(evals, stake=stake)
        n_matches = len(evals)
        st.caption(f"Simulation basee sur {n_matches} matchs deja evalues "
                   "(mêmes matchs que l'onglet Fiabilite de l'IA).")

        def show_recap(df, title, caption):
            st.subheader(title)
            st.caption(caption)
            fmt = df.copy()
            fmt["Taux de reussite"] = (fmt["Taux de reussite"] * 100).round(1).astype(str) + " %"
            fmt["ROI (%)"] = fmt["ROI (%)"].round(1).astype(str) + " %"
            for col in ["Cote moyenne jouee", "Cote moyenne (paris gagnes)"]:
                fmt[col] = fmt[col].round(2)
            for col in ["Mise totale (EUR)", "Total retourne (EUR)", "Profit / Perte (EUR)"]:
                fmt[col] = fmt[col].round(2)
            st.dataframe(fmt, hide_index=True, use_container_width=True)

            c1, c2 = st.columns(2)
            best = df.sort_values("ROI (%)", ascending=False).iloc[0]
            c1.metric(f"Meilleur ROI — {best['Modele']}", f"{best['ROI (%)']:.1f} %")
            c2.metric(f"Profit/Perte sur {n_matches} matchs a {stake:.0f}EUR/match",
                       f"{best['Profit / Perte (EUR)']:+.2f} EUR")

        show_recap(
            bets["recap_pick1"],
            "1) Recapitulatif — Choix 1 de l'IA (1, N ou 2)",
            "Bilan si on avait parie sur l'issue la plus probable de l'IA "
            "(pari simple 1/N/2), a chaque match, avec les 2 modeles.",
        )
        st.divider()
        show_recap(
            bets["recap_pickdc"],
            "2) Recapitulatif — Choix 1+2 de l'IA (double chance 1X / X2 / 12)",
            "Bilan si on avait parie sur la double chance couvrant les 2 "
            "issues les plus probables de l'IA a chaque match (cote plus "
            "faible, mais probabilite de gain plus elevee).",
        )

        st.divider()
        st.subheader("3) Detail par match, par modele et par choix")
        st.caption(
            "Pour chaque modele (Poisson / logistique) et chaque strategie "
            "(Choix 1 / Choix 1+2) : le pari joue, sa cote equitable, et "
            "s'il aurait ete gagnant."
        )

        detail = bets["detail"].copy()
        detail["date"] = pd.to_datetime(detail["date"]).dt.strftime("%Y-%m-%d")

        model_tab_labels = {"poisson": "🔮 Modele Poisson", "logit": "📈 Modele logistique"}
        bet_tab_labels = {"pick1": "Choix 1 (simple)", "pickdc": "Choix 1+2 (double chance)"}

        for model_key, model_label in model_tab_labels.items():
            st.markdown(f"**{model_label}**")
            sub_tabs = st.tabs(list(bet_tab_labels.values()))
            for (bet_key, bet_label), sub_tab in zip(bet_tab_labels.items(), sub_tabs):
                with sub_tab:
                    cols = ["date", "home_team", "away_team", "actual_score", "actual_outcome",
                            f"{model_key}_{bet_key}_bet", f"{model_key}_{bet_key}_odds",
                            f"{model_key}_{bet_key}_win", "source"]
                    tbl = detail[cols].copy()
                    tbl[f"{model_key}_{bet_key}_odds"] = tbl[f"{model_key}_{bet_key}_odds"].round(2)
                    tbl["gain_net_eur"] = tbl.apply(
                        lambda r: stake * (r[f"{model_key}_{bet_key}_odds"] - 1)
                        if r[f"{model_key}_{bet_key}_win"] else -stake, axis=1
                    ).round(2)
                    tbl[f"{model_key}_{bet_key}_win"] = tbl[f"{model_key}_{bet_key}_win"].map(
                        {True: "✅ Gagne", False: "❌ Perdu"}
                    )
                    tbl["source"] = tbl["source"].map({"dataset": "Dataset", "manual": "Saisie manuelle"})
                    tbl.columns = ["Date", "Domicile", "Exterieur", "Score reel", "Issue reelle",
                                   "Pari joue", "Cote", "Resultat", "Source", f"Gain net ({stake:.0f}EUR)"]
                    st.dataframe(tbl.sort_values("Date", ascending=False),
                                 hide_index=True, use_container_width=True)


# ==========================================================================
# PAGE : COTES REELLES (OddsPortal - saisie manuelle)
# ==========================================================================
elif page == "🌐 Cotes reelles (OddsPortal)":
    st.title("Paris sportifs avec les vraies cotes du marche (OddsPortal)")
    st.info(
        "OddsPortal interdit le scraping automatise dans ses CGU et protege "
        "son site contre les robots (Cloudflare, contenu charge en JS) : "
        "cette page ne se connecte donc pas au site automatiquement. "
        "La marche a suivre : va sur **oddsportal.com**, cherche le match, "
        "releve les cotes fermees (closing odds) du marche **1N2**, puis "
        "colle-les ci-dessous. Les cotes Double Chance (1X/X2/12) sont "
        "facultatives : si tu ne les saisis pas, elles sont estimees a "
        "partir des cotes 1N2."
    )

    stake_real = st.number_input(
        "Mise par match simulee (EUR)", min_value=1.0, max_value=1000.0,
        value=pt.STAKE_DEFAULT, step=5.0, key="stake_real_odds",
    )

    with st.spinner("Chargement des matchs deja evalues..."):
        evals = pt.evaluate_predictions(store)

    if len(evals) == 0:
        st.warning("Pas encore de matchs evalues (voir l'onglet Fiabilite de l'IA).")
        st.stop()

    odds_df = om.load_manual_odds(".")
    n_with_odds = len(odds_df)
    st.metric("Matchs avec cotes reelles saisies", f"{n_with_odds} / {len(evals)}")

    # ---------------------------------------------------------------- #
    # FORMULAIRE DE SAISIE
    # ---------------------------------------------------------------- #
    st.subheader("Saisir / corriger les cotes d'un match")

    evals_sorted = evals.sort_values("date")
    match_labels = {
        int(r["match_id"]): f"#{int(r['match_id'])} - {r['date']} - "
                             f"{r['home_team']} vs {r['away_team']} "
                             f"({r['actual_score']})"
        for _, r in evals_sorted.iterrows()
    }
    already_entered = set(odds_df["match_id"]) if len(odds_df) else set()

    def _label_with_flag(mid):
        flag = "✅ " if mid in already_entered else "⬜ "
        return flag + match_labels[mid]

    selected_mid = st.selectbox(
        "Match", options=list(match_labels.keys()),
        format_func=_label_with_flag,
    )

    existing = om.get_odds_for_match(selected_mid, ".")

    with st.form("real_odds_form"):
        c1, c2, c3 = st.columns(3)
        odds_1 = c1.number_input("Cote 1 (victoire domicile)", min_value=1.01, step=0.05,
                                  value=float(existing["odds_1"]) if existing is not None else 2.00)
        odds_N = c2.number_input("Cote N (nul)", min_value=1.01, step=0.05,
                                  value=float(existing["odds_N"]) if existing is not None else 3.30)
        odds_2 = c3.number_input("Cote 2 (victoire exterieur)", min_value=1.01, step=0.05,
                                  value=float(existing["odds_2"]) if existing is not None else 3.50)

        st.caption("Double chance (facultatif - laisser a 0 pour estimation automatique)")
        c4, c5, c6 = st.columns(3)
        odds_1X = c4.number_input("Cote 1X", min_value=0.0, step=0.05,
                                   value=float(existing["odds_1X"]) if existing is not None else 0.0)
        odds_X2 = c5.number_input("Cote X2", min_value=0.0, step=0.05,
                                   value=float(existing["odds_X2"]) if existing is not None else 0.0)
        odds_12 = c6.number_input("Cote 12", min_value=0.0, step=0.05,
                                   value=float(existing["odds_12"]) if existing is not None else 0.0)

        bookmaker = st.text_input(
            "Bookmaker (optionnel)",
            value=str(existing["bookmaker"]) if existing is not None and pd.notna(existing["bookmaker"]) else "",
        )

        submitted = st.form_submit_button("Enregistrer les cotes de ce match")
        if submitted:
            ok, msg = om.save_manual_odds(
                selected_mid, odds_1, odds_N, odds_2,
                odds_1X=odds_1X or None, odds_X2=odds_X2 or None, odds_12=odds_12 or None,
                bookmaker=bookmaker, data_dir=".",
            )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    if existing is not None:
        if st.button("🗑️ Supprimer les cotes de ce match"):
            ok, msg = om.delete_manual_odds(selected_mid, ".")
            (st.success if ok else st.error)(msg)
            st.rerun()

    # ---------------------------------------------------------------- #
    # MATCHS ENCORE SANS COTES
    # ---------------------------------------------------------------- #
    pending = om.matches_missing_real_odds(evals, ".")
    if len(pending):
        with st.expander(f"📋 {len(pending)} match(s) sans cotes reelles saisies"):
            p = pending[["match_id", "date", "home_team", "away_team", "actual_score"]].copy()
            p.columns = ["Match ID", "Date", "Domicile", "Exterieur", "Score reel"]
            st.dataframe(p, hide_index=True, use_container_width=True)

    # ---------------------------------------------------------------- #
    # RECAPITULATIFS AVEC LES VRAIES COTES
    # ---------------------------------------------------------------- #
    if n_with_odds == 0:
        st.info("Saisis au moins une cote reelle ci-dessus pour voir apparaitre les recapitulatifs.")
        st.stop()

    st.divider()
    st.subheader("Recapitulatifs — vraies cotes du marche")
    st.caption(
        "Meme principe que l'onglet 'Paris sportifs', mais avec les cotes "
        "reelles saisies manuellement au lieu des cotes equitables des "
        "modeles. Uniquement calcule sur les matchs pour lesquels une cote "
        "reelle a ete saisie."
    )

    model_labels = {"poisson": "Modele Poisson (score exact)", "logit": "Modele logistique (issue)"}
    bet_labels = {"pick1": "Choix 1 (issue la plus probable)", "pickdc": "Choix 1+2 (double chance)"}

    def build_real_recap(bet_kind: str) -> pd.DataFrame:
        rows = []
        for model_key, label in model_labels.items():
            summ = om.summarize_real_odds_betting(evals, model_key, bet_kind, stake_real, data_dir=".")
            if summ.get("n_bets", 0) == 0:
                rows.append({"Modele": label, "Paris joues": 0})
                continue
            rows.append({
                "Modele": label,
                "Paris joues": summ["n_bets"],
                "Paris gagnes": summ["n_win"],
                "Taux de reussite": f"{summ['win_rate']*100:.1f} %",
                "Cote moyenne jouee": round(summ["avg_odds_played"], 2),
                "Mise totale (EUR)": round(summ["total_staked"], 2),
                "Total retourne (EUR)": round(summ["total_returned"], 2),
                "Profit / Perte (EUR)": round(summ["profit"], 2),
                "ROI (%)": f"{summ['roi_pct']:.1f} %",
            })
        return pd.DataFrame(rows)

    for bet_kind, label in bet_labels.items():
        st.markdown(f"**{label}**")
        st.dataframe(build_real_recap(bet_kind), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------------- #
    # DETAIL PAR MATCH, MODELE ET CHOIX
    # ---------------------------------------------------------------- #
    st.divider()
    st.subheader("Detail par match, par modele et par choix (vraies cotes)")

    for model_key, model_label in model_labels.items():
        st.markdown(f"**{model_label}**")
        sub_tabs = st.tabs(list(bet_labels.values()))
        for (bet_key, bet_label), sub_tab in zip(bet_labels.items(), sub_tabs):
            with sub_tab:
                summ = om.summarize_real_odds_betting(evals, model_key, bet_key, stake_real, data_dir=".")
                if summ.get("n_bets", 0) == 0:
                    st.caption("Aucun match avec cotes reelles pour cette combinaison.")
                    continue
                tbl = summ["detail"].copy()
                bet_col = f"{model_key}_{bet_key}_bet"
                tbl["gain_net_eur"] = tbl.apply(
                    lambda r: stake_real * (r["real_odds_played"] - 1) if r["win"] else -stake_real,
                    axis=1,
                ).round(2)
                tbl["real_odds_played"] = tbl["real_odds_played"].round(2)
                tbl["win"] = tbl["win"].map({True: "✅ Gagne", False: "❌ Perdu"})
                tbl = tbl[["date", "home_team", "away_team", "actual_score", "actual_outcome",
                           bet_col, "real_odds_played", "win", "bookmaker", "gain_net_eur"]]
                tbl.columns = ["Date", "Domicile", "Exterieur", "Score reel", "Issue reelle",
                               "Pari joue", "Cote reelle", "Resultat", "Bookmaker",
                               f"Gain net ({stake_real:.0f}EUR)"]
                st.dataframe(tbl.sort_values("Date", ascending=False),
                             hide_index=True, use_container_width=True)


# ==========================================================================
# PAGE : CALIBRATION DES MODELES
# ==========================================================================
elif page == "📐 Calibration des modeles":
    st.title("Calibration des modeles — l'IA est-elle bien reglee ?")
    st.caption(
        "Quand un modele annonce '70% de chance de victoire', est-ce que "
        "ca arrive vraiment ~70% du temps ? Un modele bien calibre colle a "
        "la diagonale (proba annoncee = frequence observee). S'il est "
        "au-dessus, il est **trop confiant** (surestime ses chances) ; "
        "en dessous, il est **trop prudent** (sous-estime ses chances)."
    )

    n_bins = st.slider("Nombre de tranches de probabilite", min_value=5, max_value=15, value=8)

    with st.spinner("Calcul des predictions retrospectives (walk-forward)..."):
        evals = pt.evaluate_predictions(store)

    if len(evals) == 0:
        st.info("Pas encore assez de matchs joues pour evaluer la calibration.")
    else:
        calib = pt.calibration_summary(evals, n_bins=n_bins)
        st.caption(
            f"Base sur {calib['n_matches']} matchs evalues x 3 issues possibles "
            f"(Victoire domicile / Nul / Victoire exterieur) = "
            f"{calib['n_matches']*3} points de probabilite par modele."
        )

        c1, c2 = st.columns(2)
        c1.metric("Score de Brier — Modele Poisson", f"{calib['poisson']['brier']:.3f}",
                   help="Plus bas = meilleur. 0 = parfait. ~0.667 = un pronostic "
                        "'ignorant' qui donnerait toujours 33/33/33%.")
        c2.metric("Score de Brier — Modele logistique", f"{calib['logit']['brier']:.3f}",
                   help="Plus bas = meilleur. 0 = parfait. ~0.667 = un pronostic "
                        "'ignorant' qui donnerait toujours 33/33/33%.")

        st.subheader("Diagramme de fiabilite (reliability diagram)")
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                         line=dict(dash="dash", color="gray"))

        for model_key, color, label in [("poisson", "#636EFA", "Modele Poisson"),
                                          ("logit", "#EF553B", "Modele logistique")]:
            curve = calib[model_key]["curve"]
            if len(curve) == 0:
                continue
            fig.add_scatter(
                x=curve["proba_moyenne_annoncee"], y=curve["frequence_observee"],
                mode="lines+markers", name=label,
                marker=dict(size=np.clip(curve["n_observations"], 6, 30), color=color),
                line=dict(color=color),
                hovertext=[f"n={n}" for n in curve["n_observations"]],
            )

        fig.update_layout(
            xaxis_title="Probabilite moyenne annoncee par le modele",
            yaxis_title="Frequence reellement observee",
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Taille des points = nombre d'observations dans la tranche "
                   "(une tranche avec peu de points est moins fiable statistiquement).")

        st.subheader("Detail par tranche de probabilite")
        tabs = st.tabs(["🔮 Modele Poisson", "📈 Modele logistique"])
        for tab, model_key in zip(tabs, ["poisson", "logit"]):
            with tab:
                curve = calib[model_key]["curve"].copy()
                if len(curve) == 0:
                    st.caption("Pas assez de donnees.")
                    continue
                curve["proba_moyenne_annoncee"] = (curve["proba_moyenne_annoncee"] * 100).round(1)
                curve["frequence_observee"] = (curve["frequence_observee"] * 100).round(1)
                curve["ecart"] = (curve["ecart"] * 100).round(1)
                curve = curve[["bin_label", "n_observations", "proba_moyenne_annoncee",
                                "frequence_observee", "ecart"]]
                curve.columns = ["Tranche", "Nb observations", "Proba annoncee (%)",
                                  "Frequence observee (%)", "Ecart (annoncee - observee, pts)"]
                st.dataframe(curve, hide_index=True, use_container_width=True)


# ==========================================================================
# PAGE : SIMULATION MONTE CARLO DU TOURNOI
# ==========================================================================
elif page == "🌀 Simulation Monte Carlo":
    st.title("Simulation Monte Carlo du reste du tournoi")
    st.caption(
        "Simule N fois la suite du tournoi a partir de l'etat actuel du "
        "bracket : les matchs deja joues restent figes, les matchs "
        "restants sont tires au sort selon les probabilites du modele de "
        "Poisson, en propageant automatiquement les qualifications "
        "(quarts -> demies -> finale) a chaque tirage."
    )
    with st.expander("Hypotheses et limites de la simulation"):
        st.markdown(
            "- En cas de match nul en elimination directe, la probabilite "
            "de victoire est repartie 50/50 pour simuler les tirs au but "
            "(le modele ne sait pas qui est meilleur aux penaltys).\n"
            "- La forme de chaque equipe est figee a son niveau **actuel** "
            "pendant toute la simulation d'un tirage (elle n'est pas "
            "recalculee round par round), pour rester rapide.\n"
            "- Les resultats ne sont fiables que dans la mesure ou le "
            "modele sous-jacent l'est (voir l'onglet Calibration)."
        )

    c1, c2 = st.columns(2)
    n_sims = c1.select_slider(
        "Nombre de simulations", options=[500, 1000, 2000, 5000, 10000, 20000], value=5000,
    )
    seed = c2.number_input("Graine aleatoire (pour reproduire le meme tirage)",
                            min_value=0, max_value=999999, value=42, step=1)

    run = st.button("🎲 Lancer la simulation", type="primary")

    @st.cache_data(show_spinner="Simulation en cours...")
    def _run_simulation(store_version, n_sims, seed):
        result = ts.simulate_tournament(store, n_simulations=n_sims, seed=seed)
        return result

    if run or "last_sim_result" in st.session_state:
        if run:
            result = _run_simulation(_manual_results_version(), n_sims, seed)
            st.session_state["last_sim_result"] = result
        else:
            result = st.session_state["last_sim_result"]

        if "error" in result:
            st.info(result["error"])
        else:
            df = ts.summarize_simulation(result, store)
            st.success(f"{result['n_simulations']} simulations effectuees.")

            st.subheader("Probabilite de remporter le tournoi")
            top = df.head(16).copy()
            fig = px.bar(top, x="Vainqueur", y="Equipe", orientation="h",
                         labels={"Vainqueur": "Probabilite de titre"})
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                               xaxis_tickformat=".0%", height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Probabilites de qualification par stade")
            display_cols = ["Equipe"] + [c for c in
                ["Quarts", "Demies", "Finale", "Vainqueur"] if c in df.columns]
            fmt = df[display_cols].copy()
            for c in display_cols[1:]:
                fmt[c] = (fmt[c] * 100).round(1).astype(str) + " %"
            st.dataframe(fmt, hide_index=True, use_container_width=True)
    else:
        st.info("Configure les parametres puis clique sur 'Lancer la simulation'.")
