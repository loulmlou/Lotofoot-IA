"""Frontend Streamlit pour LotoFoot AI Analyst."""

import sys
import os
from datetime import date, datetime

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import STRATEGIES, BUDGETS, LOTOFOOT_TYPES
from database.connection import SessionLocal, init_db
from database.models import Equipe, Competition, GrilleLotoFoot, StatistiqueGrille
from models.predictor import Predictor
from generator.grid_generator import GridGenerator
from analysis.grid_analysis import get_distribution_by_type
from frontend.helpers import RESULT_COLORS, GRID_TYPE_CODES, color_result, format_results_html, match_team_name
from collectors.lotofoot_scraper import fetch_upcoming_grilles


def get_equipes(session):
    """Récupère la liste des équipes triées par nom."""
    return session.query(Equipe).order_by(Equipe.nom).all()


def get_competitions(session):
    """Récupère la liste des compétitions."""
    return session.query(Competition).order_by(Competition.nom).all()


# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LotoFoot AI Analyst",
    page_icon="⚽",
    layout="wide",
)

st.sidebar.title("LotoFoot AI Analyst")
page = st.sidebar.radio(
    "Navigation",
    ["Générateur de grilles", "Prédiction de match", "Historique des grilles", "Dashboard stats"],
)

# ---------------------------------------------------------------------------
# Page 1 — Générateur de grilles
# ---------------------------------------------------------------------------

if page == "Générateur de grilles":
    st.header("Générateur de grilles")

    session = SessionLocal()
    try:
        equipes = get_equipes(session)
        competitions = get_competitions(session)
    finally:
        session.close()

    noms_equipes = [e.nom for e in equipes]
    equipes_by_name = {e.nom: e for e in equipes}
    noms_competitions = [c.nom for c in competitions]
    comps_by_name = {c.nom: c for c in competitions}

    # --- Sélection du mode ---
    mode = st.radio(
        "Mode de saisie",
        ["Grille à venir (FDJ)", "Saisie manuelle"],
        horizontal=True,
    )

    # --- Paramètres communs ---
    col1, col2, col3 = st.columns(3)
    with col1:
        grid_type_keys = list(LOTOFOOT_TYPES.keys())
        grid_type_labels = [f"{v['code']} ({k})" for k, v in LOTOFOOT_TYPES.items()]
        grid_type_idx = st.selectbox(
            "Type de grille", range(len(grid_type_keys)),
            format_func=lambda i: grid_type_labels[i], index=0,
        )
        grid_type_key = grid_type_keys[grid_type_idx]
        grid_type = LOTOFOOT_TYPES[grid_type_key]["code"]
    with col2:
        strategy = st.selectbox("Stratégie", STRATEGIES, index=1)
    with col3:
        budget = st.slider("Budget (mises)", min_value=1, max_value=50, value=5)

    nb_matchs = LOTOFOOT_TYPES[grid_type_key]["nb_matchs"]

    # ===================================================================
    # Mode 1 — Grille à venir (FDJ)
    # ===================================================================
    if mode == "Grille à venir (FDJ)":
        # Bouton de recherche
        if st.button("Chercher les grilles à venir", type="secondary"):
            with st.spinner("Recherche des grilles à venir sur le site FDJ..."):
                found = fetch_upcoming_grilles(grille_type=grid_type_key)
            st.session_state["upcoming_grilles"] = found
            if not found:
                st.warning("Aucune grille à venir trouvée pour ce type.")

        upcoming = st.session_state.get("upcoming_grilles", [])

        if upcoming:
            # Sélection de la grille
            grille_labels = [
                f"N°{g['numero']} — {g['date'] or '?'} — {len(g['matchs'])} matchs"
                for g in upcoming
            ]
            selected_idx = st.selectbox("Grille à venir", range(len(upcoming)),
                                        format_func=lambda i: grille_labels[i])
            selected_grille = upcoming[selected_idx]

            st.subheader(f"Matchs de la grille N°{selected_grille['numero']}")

            # Préremplir les matchs avec matching d'équipes
            matchs_input = []
            has_unmatched = False
            for i, m in enumerate(selected_grille["matchs"]):
                fdj_dom = m["domicile"]
                fdj_ext = m["exterieur"]
                matched_dom = match_team_name(fdj_dom, equipes)
                matched_ext = match_team_name(fdj_ext, equipes)

                with st.expander(f"Match {i + 1} : {fdj_dom} - {fdj_ext}", expanded=True):
                    cols = st.columns([3, 3, 3, 2])
                    with cols[0]:
                        if matched_dom:
                            dom_idx = noms_equipes.index(matched_dom.nom)
                        else:
                            dom_idx = 0
                            has_unmatched = True
                            st.caption(f"Non reconnu : « {fdj_dom} »")
                        dom = st.selectbox("Domicile", noms_equipes, index=dom_idx, key=f"fdj_dom_{i}")
                    with cols[1]:
                        if matched_ext:
                            ext_idx = noms_equipes.index(matched_ext.nom)
                        else:
                            ext_idx = 0
                            has_unmatched = True
                            st.caption(f"Non reconnu : « {fdj_ext} »")
                        ext = st.selectbox("Extérieur", noms_equipes, index=ext_idx, key=f"fdj_ext_{i}")
                    with cols[2]:
                        comp = st.selectbox("Compétition", noms_competitions, key=f"fdj_comp_{i}")
                    with cols[3]:
                        match_date = st.date_input(
                            "Date",
                            value=selected_grille.get("date") or date.today(),
                            key=f"fdj_date_{i}",
                        )
                    matchs_input.append({
                        "dom": dom, "ext": ext, "comp": comp, "date": match_date,
                    })

            if has_unmatched:
                st.info("Certaines équipes n'ont pas été reconnues automatiquement. "
                        "Veuillez les sélectionner manuellement ci-dessus.")

            # Bouton générer
            if st.button("Générer les grilles", type="primary", key="gen_fdj"):
                if not noms_equipes or not noms_competitions:
                    st.warning("Aucune équipe ou compétition en base. Importez d'abord des données.")
                else:
                    with st.spinner("Génération en cours..."):
                        predictor = Predictor(strategy=strategy)
                        generator = GridGenerator(predictor=predictor, strategy=strategy)

                        session = SessionLocal()
                        try:
                            matches_features = []
                            for m_input in matchs_input:
                                eq_dom = equipes_by_name[m_input["dom"]]
                                eq_ext = equipes_by_name[m_input["ext"]]
                                comp_obj = comps_by_name[m_input["comp"]]
                                m_date = datetime.combine(m_input["date"], datetime.min.time())

                                pred = predictor.predict_from_ids(
                                    eq_dom.id, eq_ext.id, comp_obj.id, m_date, session=session
                                )
                                pred["equipe_dom"] = m_input["dom"]
                                pred["equipe_ext"] = m_input["ext"]
                                matches_features.append(pred)
                        finally:
                            session.close()

                        grilles = generator.generate(matches_features, grid_type=grid_type, budget=budget)

                    if not grilles:
                        st.warning("Aucune grille générée.")
                    else:
                        st.success(f"{len(grilles)} grille(s) générée(s)")
                        for idx, grille in enumerate(grilles):
                            resultats = grille.get("resultats", "")
                            confiance = grille.get("confiance", 0)
                            probabilite = grille.get("probabilite", 0)

                            col_a, col_b, col_c = st.columns([4, 2, 2])
                            with col_a:
                                st.markdown(
                                    f"**Grille {idx + 1}** : {format_results_html(resultats)}",
                                    unsafe_allow_html=True,
                                )
                            with col_b:
                                st.metric("Confiance", f"{confiance:.1%}")
                            with col_c:
                                st.metric("Probabilité", f"{probabilite:.4%}")

                            matchs_detail = grille.get("matchs", [])
                            if matchs_detail:
                                with st.expander(f"Détails grille {idx + 1}"):
                                    rows = []
                                    for j, md in enumerate(matchs_detail):
                                        rows.append({
                                            "#": j + 1,
                                            "Match": f"{matchs_input[j]['dom']} - {matchs_input[j]['ext']}",
                                            "Résultat": md.get("prediction", resultats[j] if j < len(resultats) else "?"),
                                            "P(1)": f"{md.get('prob_1', 0):.1%}",
                                            "P(N)": f"{md.get('prob_n', 0):.1%}",
                                            "P(2)": f"{md.get('prob_2', 0):.1%}",
                                            "Confiance": f"{md.get('confiance', 0):.1%}",
                                        })
                                    st.table(pd.DataFrame(rows))

    # ===================================================================
    # Mode 2 — Saisie manuelle
    # ===================================================================
    else:
        st.subheader(f"Saisie des {nb_matchs} matchs")

        matchs_input = []
        for i in range(nb_matchs):
            with st.expander(f"Match {i + 1}", expanded=(i < 3)):
                cols = st.columns([3, 3, 3, 2])
                with cols[0]:
                    dom = st.selectbox("Domicile", noms_equipes, key=f"dom_{i}")
                with cols[1]:
                    ext = st.selectbox("Extérieur", noms_equipes, key=f"ext_{i}")
                with cols[2]:
                    comp = st.selectbox("Compétition", noms_competitions, key=f"comp_{i}")
                with cols[3]:
                    match_date = st.date_input("Date", value=date.today(), key=f"date_{i}")
                matchs_input.append({
                    "dom": dom, "ext": ext, "comp": comp, "date": match_date,
                })

        if st.button("Générer les grilles", type="primary", key="gen_manual"):
            if not noms_equipes or not noms_competitions:
                st.warning("Aucune équipe ou compétition en base. Importez d'abord des données.")
            else:
                with st.spinner("Génération en cours..."):
                    predictor = Predictor(strategy=strategy)
                    generator = GridGenerator(predictor=predictor, strategy=strategy)

                    session = SessionLocal()
                    try:
                        matches_features = []
                        for m in matchs_input:
                            eq_dom = equipes_by_name[m["dom"]]
                            eq_ext = equipes_by_name[m["ext"]]
                            comp_obj = comps_by_name[m["comp"]]
                            match_date = datetime.combine(m["date"], datetime.min.time())

                            pred = predictor.predict_from_ids(
                                eq_dom.id, eq_ext.id, comp_obj.id, match_date, session=session
                            )
                            pred["equipe_dom"] = m["dom"]
                            pred["equipe_ext"] = m["ext"]
                            matches_features.append(pred)
                    finally:
                        session.close()

                    grilles = generator.generate(matches_features, grid_type=grid_type, budget=budget)

                if not grilles:
                    st.warning("Aucune grille générée.")
                else:
                    st.success(f"{len(grilles)} grille(s) générée(s)")
                    for idx, grille in enumerate(grilles):
                        resultats = grille.get("resultats", "")
                        confiance = grille.get("confiance", 0)
                        probabilite = grille.get("probabilite", 0)

                        col_a, col_b, col_c = st.columns([4, 2, 2])
                        with col_a:
                            st.markdown(
                                f"**Grille {idx + 1}** : {format_results_html(resultats)}",
                                unsafe_allow_html=True,
                            )
                        with col_b:
                            st.metric("Confiance", f"{confiance:.1%}")
                        with col_c:
                            st.metric("Probabilité", f"{probabilite:.4%}")

                        matchs_detail = grille.get("matchs", [])
                        if matchs_detail:
                            with st.expander(f"Détails grille {idx + 1}"):
                                rows = []
                                for j, md in enumerate(matchs_detail):
                                    rows.append({
                                        "#": j + 1,
                                        "Match": f"{matchs_input[j]['dom']} - {matchs_input[j]['ext']}",
                                        "Résultat": md.get("prediction", resultats[j] if j < len(resultats) else "?"),
                                        "P(1)": f"{md.get('prob_1', 0):.1%}",
                                        "P(N)": f"{md.get('prob_n', 0):.1%}",
                                        "P(2)": f"{md.get('prob_2', 0):.1%}",
                                        "Confiance": f"{md.get('confiance', 0):.1%}",
                                    })
                                st.table(pd.DataFrame(rows))

# ---------------------------------------------------------------------------
# Page 2 — Prédiction de match
# ---------------------------------------------------------------------------

elif page == "Prédiction de match":
    st.header("Prédiction de match")

    session = SessionLocal()
    try:
        equipes = get_equipes(session)
        competitions = get_competitions(session)
    finally:
        session.close()

    noms_equipes = [e.nom for e in equipes]
    equipes_by_name = {e.nom: e for e in equipes}
    noms_competitions = [c.nom for c in competitions]
    comps_by_name = {c.nom: c for c in competitions}

    col1, col2 = st.columns(2)
    with col1:
        dom = st.selectbox("Équipe domicile", noms_equipes)
    with col2:
        ext = st.selectbox("Équipe extérieur", noms_equipes)

    col3, col4 = st.columns(2)
    with col3:
        comp = st.selectbox("Compétition", noms_competitions)
    with col4:
        match_date = st.date_input("Date du match", value=date.today())

    if st.button("Prédire", type="primary"):
        if not noms_equipes or not noms_competitions:
            st.warning("Aucune équipe ou compétition en base.")
        else:
            with st.spinner("Calcul des probabilités..."):
                predictor = Predictor(strategy="equilibree")
                session = SessionLocal()
                try:
                    eq_dom = equipes_by_name[dom]
                    eq_ext = equipes_by_name[ext]
                    comp_obj = comps_by_name[comp]
                    dt = datetime.combine(match_date, datetime.min.time())

                    result = predictor.predict_from_ids(
                        eq_dom.id, eq_ext.id, comp_obj.id, dt, session=session
                    )
                finally:
                    session.close()

            prob_1 = result.get("prob_1", 0.33)
            prob_n = result.get("prob_n", 0.33)
            prob_2 = result.get("prob_2", 0.34)
            prediction = result.get("prediction", "?")
            confiance = result.get("confiance", 0)

            st.subheader(f"Prédiction : {prediction}")
            st.metric("Confiance", f"{confiance:.1%}")

            # Barres de probabilité
            bar_data = pd.DataFrame({
                "Résultat": ["1 (Dom)", "N (Nul)", "2 (Ext)"],
                "Probabilité": [prob_1, prob_n, prob_2],
            })
            fig_bar = px.bar(
                bar_data,
                x="Résultat",
                y="Probabilité",
                color="Résultat",
                color_discrete_map={
                    "1 (Dom)": RESULT_COLORS["1"],
                    "N (Nul)": RESULT_COLORS["N"],
                    "2 (Ext)": RESULT_COLORS["2"],
                },
                text_auto=".1%",
            )
            fig_bar.update_layout(showlegend=False, yaxis_tickformat=".0%")

            # Camembert
            fig_pie = px.pie(
                bar_data,
                names="Résultat",
                values="Probabilité",
                color="Résultat",
                color_discrete_map={
                    "1 (Dom)": RESULT_COLORS["1"],
                    "N (Nul)": RESULT_COLORS["N"],
                    "2 (Ext)": RESULT_COLORS["2"],
                },
                hole=0.3,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_b:
                st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 3 — Historique des grilles
# ---------------------------------------------------------------------------

elif page == "Historique des grilles":
    st.header("Historique des grilles")

    grid_codes = list(GRID_TYPE_CODES.keys())
    filtre_type = st.selectbox("Filtrer par type", ["Tous"] + grid_codes)

    session = SessionLocal()
    try:
        query = session.query(GrilleLotoFoot, StatistiqueGrille).outerjoin(
            StatistiqueGrille, GrilleLotoFoot.id == StatistiqueGrille.grille_id
        ).order_by(GrilleLotoFoot.date.desc())

        if filtre_type != "Tous":
            query = query.filter(GrilleLotoFoot.type_grille == filtre_type)

        rows = query.limit(200).all()

        if not rows:
            st.info("Aucune grille trouvée.")
        else:
            data = []
            for grille, stats in rows:
                row = {
                    "Date": grille.date,
                    "Type": grille.type_grille,
                    "Résultats": grille.resultats or "",
                    "Rapport Rang 1": f"{grille.rapport_rang1:.2f}" if grille.rapport_rang1 else "-",
                    "Gagnants Rang 1": grille.nombre_gagnants_rang1 if grille.nombre_gagnants_rang1 is not None else "-",
                }
                if stats:
                    row["Profil"] = stats.profil or "-"
                    row["Entropie"] = f"{stats.entropie:.3f}" if stats.entropie is not None else "-"
                    row["Chaos"] = f"{stats.indice_chaos:.3f}" if stats.indice_chaos is not None else "-"
                else:
                    row["Profil"] = "-"
                    row["Entropie"] = "-"
                    row["Chaos"] = "-"
                data.append(row)

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.caption(f"{len(data)} grille(s) affichée(s)")
    finally:
        session.close()

# ---------------------------------------------------------------------------
# Page 4 — Dashboard stats
# ---------------------------------------------------------------------------

elif page == "Dashboard stats":
    st.header("Dashboard statistiques")

    dist = get_distribution_by_type()

    if not dist:
        st.info("Aucune statistique disponible. Lancez d'abord l'analyse des grilles.")
    else:
        # Nombre de grilles par type
        types = list(dist.keys())
        nb_grilles = [dist[t].get("nb_grilles", 0) for t in types]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Nombre de grilles par type")
            fig_count = px.bar(
                x=types,
                y=nb_grilles,
                labels={"x": "Type", "y": "Nombre"},
                color=types,
            )
            st.plotly_chart(fig_count, use_container_width=True)

        with col2:
            st.subheader("Distribution 1/N/2 par type")
            bar_data = []
            for t in types:
                bar_data.append({"Type": t, "Résultat": "1", "Moyenne": dist[t].get("moy_1", 0)})
                bar_data.append({"Type": t, "Résultat": "N", "Moyenne": dist[t].get("moy_n", 0)})
                bar_data.append({"Type": t, "Résultat": "2", "Moyenne": dist[t].get("moy_2", 0)})

            fig_dist = px.bar(
                pd.DataFrame(bar_data),
                x="Type",
                y="Moyenne",
                color="Résultat",
                barmode="group",
                color_discrete_map=RESULT_COLORS,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Stats moyennes
        st.subheader("Statistiques moyennes par type")
        stats_rows = []
        for t in types:
            d = dist[t]
            stats_rows.append({
                "Type": t,
                "Grilles": d.get("nb_grilles", 0),
                "Moy. 1": f"{d.get('moy_1', 0):.2f}",
                "Moy. N": f"{d.get('moy_n', 0):.2f}",
                "Moy. 2": f"{d.get('moy_2', 0):.2f}",
                "Entropie": f"{d.get('moy_entropie', 0):.3f}",
                "Chaos": f"{d.get('moy_chaos', 0):.3f}",
                "Alternance": f"{d.get('moy_alternance', 0):.3f}",
            })
        st.table(pd.DataFrame(stats_rows))
