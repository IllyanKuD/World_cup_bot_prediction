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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fifa_2026_app import DataStore, predict_match, poisson_pmf

st.set_page_config(page_title="FIFA World Cup 2026 - Analytics", layout="wide",
                    page_icon="⚽")


# ==========================================================================
# CHARGEMENT (mis en cache : les modeles ne sont entraines qu'une fois)
# ==========================================================================
@st.cache_resource(show_spinner="Chargement des donnees et entrainement des modeles...")
def load_store():
    return DataStore(".")


try:
    store = load_store()
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
     "🔮 Prediction"],
)

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
