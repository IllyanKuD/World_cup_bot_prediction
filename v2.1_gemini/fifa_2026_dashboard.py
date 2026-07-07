#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 FIFA WORLD CUP 2026 - DASHBOARD GRAPHIQUE & ENGIN DE SUIVI (v3)
==============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fifa_2026_app import DataStore, predict_match, poisson_pmf

st.set_page_config(page_title="FIFA World Cup 2026 - Analytics & Bracket", layout="wide", page_icon="⚽")

MANUAL_DATA_FILE = "fifa_2026_manual_data.json"

# ==========================================================================
# GESTION DE LA PERSISTANCE (JSON LOCAL)
# ==========================================================================
def load_manual_data():
    if os.path.exists(MANUAL_DATA_FILE):
        with open(MANUAL_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"matches": {}, "bracket_teams": {}}

def save_manual_data(data):
    with open(MANUAL_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Structure de chaînage du Bracket KO (Exemple basé sur l'ID de match)
# Associe le match actuel au match suivant dans l'arbre et définit le slot ('home' ou 'away')
KNOCKOUT_PROGRESSION = {
    # Huitièmes (IDs fictifs illustratifs à adapter selon votre dataset)
    97: {"next": 105, "slot": "home"},
    98: {"next": 105, "slot": "away"},
    99: {"next": 106, "slot": "home"},
    100: {"next": 106, "slot": "away"},
    101: {"next": 107, "slot": "home"},
    102: {"next": 107, "slot": "away"},
    103: {"next": 108, "slot": "home"},
    104: {"next": 108, "slot": "away"},
    # Quarts
    105: {"next": 109, "slot": "home"},
    106: {"next": 109, "slot": "away"},
    107: {"next": 110, "slot": "home"},
    108: {"next": 110, "slot": "away"},
    # Demis
    109: {"next": 112, "slot": "home"}, # Finale
    110: {"next": 112, "slot": "away"},
}

# ==========================================================================
# CHARGEMENT ET INJECTION DES DONNÉES MANUELLES
# ==========================================================================
@st.cache_resource(show_spinner="Chargement des données et ré-entraînement de l'IA...")
def load_store_v3():
    store = DataStore(".")
    manual_data = load_manual_data()
    
    # 1. Injection des équipes déterminées dynamiquement dans le bracket
    for mid_str, teams in manual_data.get("bracket_teams", {}).items():
        mid = int(mid_str)
        idx = store.matches[store.matches["match_id"] == mid].index
        if len(idx) > 0:
            if "home_team_id" in teams:
                store.matches.at[idx[0], "home_team_id"] = teams["home_team_id"]
            if "away_team_id" in teams:
                store.matches.at[idx[0], "away_team_id"] = teams["away_team_id"]

    # 2. Injection des scores et états complétés manuellement
    for mid_str, m_info in manual_data.get("matches", {}).items():
        mid = int(mid_str)
        idx = store.matches[store.matches["match_id"] == mid].index
        if len(idx) > 0:
            store.matches.at[idx[0], "home_score"] = m_info["home_score"]
            store.matches.at[idx[0], "away_score"] = m_info["away_score"]
            store.matches.at[idx[0], "status"] = "Completed"
            store.matches.at[idx[0], "home_xg"] = m_info.get("home_xg", 1.5)
            store.matches.at[idx[0], "away_xg"] = m_info.get("away_xg", 1.2)
            
            # Injection des buteurs dans store.events
            for scorer in m_info.get("scorers", []):
                new_event = pd.DataFrame([{
                    "match_id": mid,
                    "team_id": scorer["team_id"],
                    "player_id": scorer["player_id"],
                    "minute": scorer["minute"],
                    "event_type": "Goal"
                }])
                store.events = pd.concat([store.events, new_event], ignore_index=True)

    # Re-génération des formats longs et métriques internes pour que l'IA apprenne des scores saisis
    store._long = store._build_long_format()
    store._add_rolling_form(store._long)
    store.poisson_model, store.poisson_scaler, store.poisson_features = store._train_poisson_model()
    store.logit_model = store._train_logit_model()
    
    return store

store = load_store_v3()
TEAM_NAMES = sorted(store.teams["team_name"].tolist())

def team_id_of(name: str):
    return store.find_team_id(name)

# ==========================================================================
# SIDEBAR
# ==========================================================================
st.sidebar.title("⚽ World Cup 2026")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Vue d'ensemble", "📊 Classements de groupes", "🏳️ Statistiques equipe",
     "🧑 Statistiques joueur", "📅 Matchs d'un jour", "🔍 Detail d'un match",
     "🔮 Prediction", "✍️ Saisie Résultats & Bracket", "🎯 Validation Performance IA"],
)

n_played = int((store.matches["status"] == "Completed").sum())
n_total = len(store.matches)
st.sidebar.markdown("---")
st.sidebar.metric("Matchs joués (Total)", f"{n_played} / {n_total}")

# ==========================================================================
# PAGES EXISTANTES SIMPLIFIÉES (CONSERVÉES DE LA V2)
# ==========================================================================
if page == "🏠 Vue d'ensemble":
    st.title("Vue d'ensemble de la compétition")
    completed = store.matches[store.matches["status"] == "Completed"]
    total_goals = completed["home_score"].sum() + completed["away_score"].sum()
    avg_goals = total_goals / len(completed) if len(completed) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matchs joués", n_played)
    c2.metric("Buts marqués", f"{total_goals:.0f}")
    c3.metric("Buts / match", f"{avg_goals:.2f}")
    c4.metric("Matchs à venir", n_total - n_played)

    st.subheader("Meilleurs buteurs du tournoi")
    if len(store.events[store.events["event_type"] == "Goal"]) > 0:
        goals_series = store.events[store.events["event_type"] == "Goal"].groupby("player_id").size().reset_index(name="goals")
        goals_series = goals_series.merge(store.squads, on="player_id", how="left")
        goals_series["team"] = goals_series["team_id"].map(store.teams["team_name"])
        top_scorers = goals_series.sort_values("goals", ascending=False).head(10)
        fig = px.bar(top_scorers, x="goals", y="player_name", color="team", orientation="h", title="Top 10 Buteurs Actuels")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun but enregistré pour le moment.")

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

elif page == "🏳️ Statistiques equipe":
    st.title("Statistiques d'une équipe")
    team_name = st.selectbox("Choisir une équipe", TEAM_NAMES)
    tid = team_id_of(team_name)
    t = store.teams.loc[tid]
    rows = store._long[store._long["team_id"] == tid].sort_values("date")
    
    st.subheader(f"Historique de l'équipe : {team_name}")
    st.dataframe(rows[["date", "goals_for", "goals_against", "xg_for", "xg_against"]], use_container_width=True)

elif page == "🧑 Statistiques joueur":
    st.title("Statistiques d'un joueur")
    player_name = st.selectbox("Choisir un joueur", sorted(store.squads["player_name"].tolist()))
    p_info = store.squads[store.squads["player_name"] == player_name].iloc[0]
    st.write(p_info)

elif page == "📅 Matchs d'un jour":
    st.title("Matchs d'un jour donné")
    chosen_date = st.date_input("Choisir une date", value=store.matches["date"].min().date())
    day_matches = store.matches[store.matches["date"] == pd.to_datetime(chosen_date)]
    st.dataframe(day_matches[["match_id", "status", "home_team_id", "away_team_id", "home_score", "away_score"]], use_container_width=True)

elif page == "🔍 Detail d'un match":
    st.title("Détail complet d'un match")
    m_ids = store.matches["match_id"].tolist()
    chosen_mid = st.selectbox("Choisir le numéro de match", m_ids)
    m = store.matches[store.matches["match_id"] == chosen_mid].iloc[0]
    st.write(m)

elif page == "🔮 Prediction":
    st.title("Prédire un match")
    c1, c2, c3 = st.columns([2, 2, 1])
    team_a = c1.selectbox("Équipe A", TEAM_NAMES, index=0)
    team_b = c2.selectbox("Équipe B", TEAM_NAMES, index=1)
    neutral = c3.checkbox("Terrain neutre", value=True)
    if team_a != team_b:
        pred = predict_match(store, team_id_of(team_a), team_id_of(team_b), neutral=neutral)
        st.metric("Score le plus probable", f"{pred.best_score[0]} - {pred.best_score[1]}")

# ==========================================================================
# NOUVELLE PAGE : SAISIE DES RÉSULTATS & CONFIGURATION BRACKET
# ==========================================================================
elif page == "✍️ Saisie Résultats & Bracket":
    st.title("✍️ Saisie Manuelle des Matchs & Évolution de la Phase Finale")
    
    # Sélection du match non terminé
    pending_matches = store.matches[store.matches["status"] != "Completed"].copy()
    if len(pending_matches) == 0:
        st.success("Tous les matchs enregistrés sont complétés !")
        pending_matches = store.matches.copy()
        
    pending_matches["label"] = pending_matches.apply(
        lambda r: f"Match #{r['match_id']} | Stage ID: {r['stage_id']} | "
                  f"{store.team_name(r['home_team_id']) if pd.notna(r['home_team_id']) else 'A déterminer'} vs "
                  f"{store.team_name(r['away_team_id']) if pd.notna(r['away_team_id']) else 'A déterminer'}", axis=1
    )
    
    selected_match_label = st.selectbox("Choisir un match à renseigner :", pending_matches["label"].tolist())
    selected_mid = int(selected_match_label.split("#")[1].split(" ")[0])
    match_row = store.matches[store.matches["match_id"] == selected_mid].iloc[0]
    
    home_tid = match_row["home_team_id"]
    away_tid = match_row["away_team_id"]
    
    if pd.isna(home_tid) or pd.isna(away_tid):
        st.warning("⚠️ Les équipes de ce match ne sont pas encore définies dans l'arbre de la compétition.")
        c1, c2 = st.columns(2)
        forced_home = c1.selectbox("Forcer l'équipe Domicile :", ["-"] + TEAM_NAMES)
        forced_away = c2.selectbox("Forcer l'équipe Extérieur :", ["-"] + TEAM_NAMES)
        if st.button("Attribuer ces équipes au match"):
            m_data = load_manual_data()
            m_data["bracket_teams"][str(selected_mid)] = {
                "home_team_id": team_id_of(forced_home) if forced_home != "-" else None,
                "away_team_id": team_id_of(forced_away) if forced_away != "-" else None
            }
            save_manual_data(m_data)
            st.cache_resource.clear()
            st.rerun()
    else:
        st.subheader(f"Formulaire de score : {store.team_name(home_tid)} vs {store.team_name(away_tid)}")
        
        c1, c2 = st.columns(2)
        score_home = c1.number_input(f"Score {store.team_name(home_tid)}", min_value=0, max_value=20, value=0)
        score_away = c2.number_input(f"Score {store.team_name(away_tid)}", min_value=0, max_value=20, value=0)
        
        # Récupération des listes de joueurs pour les menus déroulants de recherche
        players_home = store.squads[store.squads["team_id"] == home_tid].sort_values("player_name")
        players_away = store.squads[store.squads["team_id"] == away_tid].sort_values("player_name")
        
        scorers_list = []
        
        if score_home > 0:
            st.markdown(f"**⚽ Buteurs - {store.team_name(home_tid)}**")
            for i in range(int(score_home)):
                p_sel = st.selectbox(f"Buteur {i+1} ({store.team_name(home_tid)})", 
                                     options=players_home["player_name"].tolist(), key=f"h_sc_{i}")
                p_id = players_home[players_home["player_name"] == p_sel].iloc[0]["player_id"]
                min_g = st.number_input(f"Minute du but {i+1}", min_value=1, max_value=120, value=30, key=f"h_min_{i}")
                scorers_list.append({"player_id": int(p_id), "team_id": int(home_tid), "minute": int(min_g)})
                
        if score_away > 0:
            st.markdown(f"**⚽ Buteurs - {store.team_name(away_tid)}**")
            for i in range(int(score_away)):
                p_sel = st.selectbox(f"Buteur {i+1} ({store.team_name(away_tid)})", 
                                     options=players_away["player_name"].tolist(), key=f"a_sc_{i}")
                p_id = players_away[players_away["player_name"] == p_sel].iloc[0]["player_id"]
                min_g = st.number_input(f"Minute du but {i+1}", min_value=1, max_value=120, value=30, key=f"a_min_{i}")
                scorers_list.append({"player_id": int(p_id), "team_id": int(away_tid), "minute": int(min_g)})

        if st.button("💾 Enregistrer et mettre à jour le Bracket"):
            m_data = load_manual_data()
            
            # Enregistrement du match actuel
            m_data["matches"][str(selected_mid)] = {
                "home_score": int(score_home),
                "away_score": int(score_away),
                "scorers": scorers_list,
                "home_xg": float(score_home) * 0.85 + 0.2,
                "away_xg": float(score_away) * 0.85 + 0.2
            }
            
            # Gestion de l'avancement automatique dans le Bracket KO
            if selected_mid in KNOCKOUT_PROGRESSION:
                nxt = KNOCKOUT_PROGRESSION[selected_mid]["next"]
                slot = KNOCKOUT_PROGRESSION[selected_mid]["slot"]
                winner_id = home_tid if score_home >= score_away else away_tid
                
                if str(nxt) not in m_data["bracket_teams"]:
                    m_data["bracket_teams"][str(nxt)] = {}
                m_data["bracket_teams"][str(nxt)][f"{slot}_team_id"] = int(winner_id)
                st.info(f"🏆 {store.team_name(winner_id)} avance au match #{nxt} ({slot}) !")
                
            save_manual_data(m_data)
            st.success("Match enregistré avec succès ! Actualisation en cours...")
            st.cache_resource.clear()
            st.rerun()

# ==========================================================================
# NOUVELLE PAGE : VALIDATION DE LA PERFORMANCE DE L'IA
# ==========================================================================
elif page == "🎯 Validation Performance IA":
    st.title("🎯 Analyse Comparative : Prédictions IA vs Résultats Réels")
    
    completed_all = store.matches[store.matches["status"] == "Completed"].copy()
    
    if len(completed_all) == 0:
        st.warning("Aucun match terminé disponible pour évaluer les modèles.")
    else:
        st.markdown("""
        Cette page confronte rétroactivement les choix de l'algorithme face aux scores observés sur le terrain. 
        Deux dimensions sont analysées : **l'issue globale (1N2)** et le **Score Exact**.
        """)
        
        outcomes_stats_poisson = {"Top 1": 0, "Top 2": 0, "Top 3": 0}
        outcomes_stats_logit = {"Top 1": 0, "Top 2": 0, "Top 3": 0}
        exact_scores_ranks = {"Top 1": 0, "Top 2": 0, "Top 3": 0, "Top 4-5": 0, "Hors Top 5": 0}
        
        for _, rm in completed_all.iterrows():
            h_id, a_id = rm["home_team_id"], rm["away_team_id"]
            if pd.isna(h_id) or pd.isna(a_id):
                continue
            
            # Calcul des prédictions rétrospectives de l'IA
            p_res = predict_match(store, h_id, a_id, neutral=True)
            
            real_hs = int(rm["home_score"])
            real_as = int(rm["away_score"])
            
            # Détermination de l'issue réelle
            if real_hs > real_as: real_out = "H"
            elif real_hs == real_as: real_out = "D"
            else: real_out = "A"
            
            # 1. Évaluation de l'issue - Modèle Poisson
            p_outcomes = [("H", p_res.proba_a), ("D", p_res.proba_draw), ("A", p_res.proba_b)]
            p_outcomes_sorted = [k for k, v in sorted(p_outcomes, key=lambda x: x[1], reverse=True)]
            
            if p_outcomes_sorted[0] == real_out: outcomes_stats_poisson["Top 1"] += 1
            elif p_outcomes_sorted[1] == real_out: outcomes_stats_poisson["Top 2"] += 1
            else: outcomes_stats_poisson["Top 3"] += 1
                
            # 2. Évaluation de l'issue - Régression Logistique
            l_outcomes_sorted = [k for k, v in sorted(p_res.logit_proba.items(), key=lambda x: x[1], reverse=True)]
            if len(l_outcomes_sorted) > 0:
                if l_outcomes_sorted[0] == real_out: outcomes_stats_logit["Top 1"] += 1
                elif l_outcomes_sorted[1] == real_out: outcomes_stats_logit["Top 2"] += 1
                else: outcomes_stats_logit["Top 3"] += 1

            # 3. Évaluation du Score Exact (Modèle Poisson Matrix)
            scores_sorted = [k for k, v in sorted(p_res.score_matrix.items(), key=lambda x: x[1], reverse=True)]
            real_score_tuple = (real_hs, real_as)
            
            if real_score_tuple == scores_sorted[0]: exact_scores_ranks["Top 1"] += 1
            elif real_score_tuple == scores_sorted[1]: exact_scores_ranks["Top 2"] += 1
            elif real_score_tuple == scores_sorted[2]: exact_scores_ranks["Top 3"] += 1
            elif real_score_tuple in scores_sorted[3:5]: exact_scores_ranks["Top 4-5"] += 1
            else: exact_scores_ranks["Hors Top 5"] += 1

        # Affichage des métriques de synthèse
        st.subheader("📊 Taux de pertinence des prédictions (En % de réussite)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🏟️ Rang de l'issue réelle (1N2)")
            df_outcomes = pd.DataFrame({
                "Classement de l'IA": list(outcomes_stats_poisson.keys()) * 2,
                "Matchs correspondants": list(outcomes_stats_poisson.values()) + list(outcomes_stats_logit.values()),
                "Modèle de calcul": ["Poisson Engine"] * 3 + ["Logit Cross-Verification"] * 3
            })
            fig_out = px.bar(df_outcomes, x="Classement de l'IA", y="Matchs correspondants", 
                             color="Modèle de calcul", barmode="group", text_auto=True,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_out, use_container_width=True)
            
        with c2:
            st.markdown("### 🥅 Précision sur le Score Exact")
            df_exact = pd.DataFrame({
                "Position du score réel": list(exact_scores_ranks.keys()),
                "Nombre d'occurrences": list(exact_scores_ranks.values())
            })
            fig_ex = px.pie(df_exact, names="Position du score réel", values="Nombre d'occurrences",
                            hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_ex, use_container_width=True)

        st.subheader("🔍 Synthèse historique des choix de l'IA")
        detailed_rows = []
        for _, rm in completed_all.iterrows():
            h_id, a_id = rm["home_team_id"], rm["away_team_id"]
            if pd.isna(h_id) or pd.isna(a_id): continue
            
            p_res = predict_match(store, h_id, a_id, neutral=True)
            real_score_str = f"{int(rm['home_score'])}-{int(rm['away_score'])}"
            ai_score_str = f"{p_res.best_score[0]}-{p_res.best_score[1]}"
            
            detailed_rows.append({
                "Match ID": rm["match_id"],
                "Domicile": store.team_name(h_id),
                "Extérieur": store.team_name(a_id),
                "Score Réel": real_score_str,
                "Score Top 1 IA": ai_score_str,
                "Proba Score Top 1": f"{p_res.score_matrix[p_res.best_score]*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(detailed_rows), hide_index=True, use_container_width=True)