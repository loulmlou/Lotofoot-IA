"""Frontend Streamlit pour LotoFoot AI Analyst — basé sur les cotes Pronosoft."""

import sys
import os
import random
from datetime import date, datetime

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import STRATEGIES, BUDGETS, LOTOFOOT_TYPES
from config.settings import ODDS_MODEL_WEIGHTS, DIFFICULTY_THRESHOLDS
from models.odds_predictor import OddsPredictor
from generator.grid_generator import GridGenerator
from collectors.pronosoft_scraper import (
    fetch_upcoming_grilles_pronosoft,
    fetch_grid_repartition,
)
from collectors.pronosoft_history import load_history, compute_grid_metrics, get_history_path
from generator.reduction_system import build_reduction_system, MAX_COMBINATIONS, RESULTS
from frontend.helpers import RESULT_COLORS, GRID_TYPE_CODES, color_result, format_results_html


# ---------------------------------------------------------------------------
# Analyse detaillee d'une grille a venir
# ---------------------------------------------------------------------------

def _compute_advanced_metrics(matches: list) -> dict:
    """Calcule les metriques avancees (HHI, marge, Gini, etc.) depuis les matchs."""
    import math
    from statistics import mean, stdev, median

    cotes_fav = []
    cotes_outsider = []
    spreads = []
    hhi_values = []
    marge_values = []
    match_categories = []  # (index, home, away, cote_fav, categorie)

    for i, m in enumerate(matches):
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        valid = [x for x in c if x > 0]
        if not valid:
            continue
        fav = min(valid)
        out = max(valid)
        cotes_fav.append(fav)
        cotes_outsider.append(out)
        spreads.append(out - fav)

        # HHI
        probs_impl = [1 / x for x in valid]
        total_prob = sum(probs_impl)
        probs_norm = [p / total_prob for p in probs_impl]
        hhi_values.append(sum(p ** 2 for p in probs_norm))

        # Marge bookmaker
        marge_values.append(total_prob - 1)

        # Categorie du match
        home = m.get("home", "?")
        away = m.get("away", "?")
        if fav < 1.5:
            cat = "Acquis"
        elif fav < 2.0:
            cat = "Intermediaire"
        else:
            cat = "Serre"
        match_categories.append((i + 1, home, away, fav, out - fav, cat))

    if not cotes_fav:
        return {}

    nb = len(cotes_fav)
    metrics = {}
    metrics["nb_matchs_avec_cotes"] = nb
    metrics["moy_cote_fav"] = mean(cotes_fav)
    metrics["std_cote_fav"] = stdev(cotes_fav) if nb > 1 else 0
    metrics["moy_spread"] = mean(spreads)
    metrics["nb_matchs_faciles"] = sum(1 for c in cotes_fav if c < 1.5)
    metrics["nb_matchs_serres"] = sum(1 for c in cotes_fav if c > 2.0)
    metrics["nb_matchs_tres_serres"] = sum(1 for c in cotes_fav if c > 2.5)
    metrics["hhi_moyen"] = mean(hhi_values)
    metrics["marge_bookmaker_moy"] = mean(marge_values)
    metrics["ratio_fav_outsider"] = mean(cotes_fav) / mean(cotes_outsider) if mean(cotes_outsider) > 0 else 0

    # Gini
    if nb > 1:
        sorted_fav = sorted(cotes_fav)
        gini_num = sum((2 * (i + 1) - nb - 1) * sorted_fav[i] for i in range(nb))
        gini_den = nb * sum(sorted_fav)
        metrics["gini_cotes_fav"] = gini_num / gini_den if gini_den > 0 else 0
    else:
        metrics["gini_cotes_fav"] = 0

    # Kelly moyen
    kelly_values = []
    for c_fav in cotes_fav:
        p_impl = 1 / c_fav
        if c_fav > 1:
            kelly_values.append((p_impl * c_fav - 1) / (c_fav - 1))
    metrics["kelly_moyen"] = mean(kelly_values) if kelly_values else 0

    # Accord cotes/joueurs
    accord_count = 0
    accord_total = 0
    for m in matches:
        c = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
        p = [m.get("pct_1", 0), m.get("pct_n", 0), m.get("pct_2", 0)]
        valid_c = [x for x in c if x > 0]
        if valid_c and any(x > 0 for x in p):
            fav_idx = c.index(min(valid_c))
            plus_joue_idx = p.index(max(p))
            accord_total += 1
            if fav_idx == plus_joue_idx:
                accord_count += 1
    metrics["accord"] = accord_count / accord_total if accord_total > 0 else 0

    # Inv spread sum
    metrics["inv_spread_sum"] = sum(1 / s if s > 0 else 10 for s in spreads)

    # Difficulte (produit cotes fav)
    metrics["difficulty"] = math.prod(cotes_fav)

    metrics["match_categories"] = match_categories
    metrics["nb_matchs"] = nb

    return metrics


def _show_detailed_analysis(matches: list, grid_type: str, grid_data: dict):
    """Affiche l'analyse detaillee d'une grille avec les metriques avancees."""
    if not matches:
        return

    adv = _compute_advanced_metrics(matches)
    if not adv:
        return

    with st.expander("Analyse detaillee de la grille", expanded=True):
        nb = adv["nb_matchs"]

        # --- GO / NO-GO ---
        st.subheader("Indicateurs GO / NO-GO")

        go_indicators = []

        # 1. Couverture cotes (NOUVEAU - signal critique)
        nb_total_matchs = len(matches)
        nb_avec_cotes = adv.get("nb_matchs_avec_cotes", nb)
        couverture = nb_avec_cotes / nb_total_matchs if nb_total_matchs else 0
        go_couverture = couverture >= 0.80
        go_indicators.append(("Couverture cotes",
                              f"{nb_avec_cotes}/{nb_total_matchs} ({couverture:.0%})",
                              go_couverture,
                              ">= 80% = GO (sinon pari aveugle)"))

        # 2. Nb pronostics (NOUVEAU - signal donnees joueurs)
        nb_pronostics = grid_data.get("nb_pronostics", 0) or 0
        go_prono = nb_pronostics >= 30
        go_indicators.append(("Nb pronostics joueurs",
                              f"{nb_pronostics}",
                              go_prono,
                              ">= 30 = GO (donnees suffisantes)"))

        # 3. Difficulte (seuil revise : moy_cote_fav au lieu du produit)
        moy_fav = adv.get("moy_cote_fav", 0)
        go_diff = moy_fav < 1.85
        go_indicators.append(("Cote moy. favori",
                              f"{moy_fav:.2f}",
                              go_diff,
                              "< 1.85 = GO (favoris solides)"))

        # 4. Marge bookmaker
        marge = adv.get("marge_bookmaker_moy", 0)
        go_marge = marge < 0.08
        go_indicators.append(("Marge bookmaker moy.", f"{marge:.1%}", go_marge,
                              "< 8% = GO"))

        # 5. Matchs serres
        serres = adv.get("nb_matchs_serres", 0)
        pct_serres = serres / nb if nb else 0
        go_serres = pct_serres < 0.30
        go_indicators.append(("Matchs serres (cote fav > 2.0)",
                              f"{serres}/{nb} ({pct_serres:.0%})", go_serres,
                              "< 30% = GO"))

        # 6. HHI moyen
        hhi = adv.get("hhi_moyen", 0)
        go_hhi = hhi > 0.40
        go_indicators.append(("HHI moyen (concentration)", f"{hhi:.3f}", go_hhi,
                              "> 0.40 = GO (matchs domines)"))

        # 7. Accord cotes/joueurs
        accord = adv.get("accord", 0)
        go_accord = accord >= 0.60
        go_indicators.append(("Accord cotes/joueurs", f"{accord:.0%}", go_accord,
                              ">= 60% = GO"))

        # 8. Std cote favori
        std_fav = adv.get("std_cote_fav", 0)
        go_std = std_fav < 0.40
        go_indicators.append(("Ecart-type cotes favori", f"{std_fav:.3f}", go_std,
                              "< 0.40 = GO (homogene)"))

        nb_go = sum(1 for _, _, g, _ in go_indicators if g)
        nb_total = len(go_indicators)

        # Afficher comme tableau colore
        go_rows = []
        for name, val, is_go, seuil in go_indicators:
            go_rows.append({
                "Indicateur": name,
                "Valeur": val,
                "Verdict": "GO" if is_go else "NO-GO",
                "Seuil": seuil,
            })

        go_df = pd.DataFrame(go_rows)
        # Colorer les verdicts
        def _color_verdict(val):
            if val == "GO":
                return "background-color: #d4edda; color: #155724"
            return "background-color: #f8d7da; color: #721c24"

        styled = go_df.style.map(
            _color_verdict, subset=["Verdict"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if nb_go >= 6:
            st.success(f"Signal fort : {nb_go}/{nb_total} GO — Grille favorable")
        elif nb_go >= 4:
            st.warning(f"Signal modere : {nb_go}/{nb_total} GO — Prudence recommandee")
        else:
            st.error(f"Signal faible : {nb_go}/{nb_total} GO — Grille defavorable, envisager NO-GO")

        # --- Metriques avancees ---
        st.subheader("Metriques avancees")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("HHI moyen", f"{adv.get('hhi_moyen', 0):.3f}",
                       help="Herfindahl-Hirschman Index : plus c'est haut, plus les matchs sont domines par un favori")
        with col2:
            st.metric("Marge bookmaker", f"{adv.get('marge_bookmaker_moy', 0):.1%}",
                       help="Marge moyenne du bookmaker par match (vig/juice)")
        with col3:
            st.metric("Kelly moyen", f"{adv.get('kelly_moyen', 0):.3f}",
                       help="Critere de Kelly fractionnel : value theorique des favoris")
        with col4:
            st.metric("Gini cotes fav", f"{adv.get('gini_cotes_fav', 0):.3f}",
                       help="Inegalite de difficulte entre matchs (0=homogene, 1=inegal)")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Spread moyen", f"{adv.get('moy_spread', 0):.2f}",
                       help="Ecart moyen entre cote favori et cote outsider")
        with col6:
            st.metric("Ratio fav/outsider", f"{adv.get('ratio_fav_outsider', 0):.3f}",
                       help="Ratio moyen cote favori / cote outsider")
        with col7:
            st.metric("Inv. spread sum", f"{adv.get('inv_spread_sum', 0):.2f}",
                       help="Somme des inverses des spreads (plus c'est haut = plus c'est serre)")
        with col8:
            st.metric("Matchs faciles", f"{adv.get('nb_matchs_faciles', 0)}/{nb}",
                       help="Matchs avec cote favori < 1.50")

        # --- Detail par match ---
        st.subheader("Classification des matchs")
        cat_rows = []
        for num, home, away, cote_fav, spread, cat in adv.get("match_categories", []):
            cat_rows.append({
                "#": num,
                "Match": f"{home} - {away}",
                "Cote fav": f"{cote_fav:.2f}",
                "Spread": f"{spread:.2f}",
                "Categorie": cat,
            })
        if cat_rows:
            cat_df = pd.DataFrame(cat_rows)

            def _color_cat(val):
                if val == "Acquis":
                    return "background-color: #d4edda; color: #155724"
                elif val == "Intermediaire":
                    return "background-color: #fff3cd; color: #856404"
                return "background-color: #f8d7da; color: #721c24"

            styled_cat = cat_df.style.map(_color_cat, subset=["Categorie"])
            st.dataframe(styled_cat, use_container_width=True, hide_index=True)

            # Resume
            n_acquis = sum(1 for r in cat_rows if r["Categorie"] == "Acquis")
            n_inter = sum(1 for r in cat_rows if r["Categorie"] == "Intermediaire")
            n_serre = sum(1 for r in cat_rows if r["Categorie"] == "Serre")
            st.caption(
                f"Acquis: {n_acquis} | Intermediaire: {n_inter} | Serre: {n_serre}"
            )


@st.cache_data(show_spinner=False)
def _cached_reduction_system(selections_tuple: tuple, guarantees_tuple: tuple) -> dict:
    """Wrapper cache autour de build_reduction_system (recalcul coûteux)."""
    selections = [list(s) for s in selections_tuple]
    guarantees = list(guarantees_tuple)
    return build_reduction_system(selections, guarantees)


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
    ["Générateur de grilles", "Système réducteur", "Historique des grilles",
     "Dashboard stats", "Suivi Bankroll", "Analyse cross-types", "Focus LF12"],
)

# ---------------------------------------------------------------------------
# Page 1 — Générateur de grilles
# ---------------------------------------------------------------------------

if page == "Générateur de grilles":
    st.header("Générateur de grilles")

    # --- Paramètres ---
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

    # Option poids optimisés
    use_optimized = st.checkbox("Utiliser les poids optimisés", value=False)

    # --- Étape 1 : Charger la grille ---
    if st.button("Charger la grille à venir", type="secondary"):
        with st.spinner("Recherche des grilles à venir sur Pronosoft..."):
            all_grilles = fetch_upcoming_grilles_pronosoft()
        st.session_state["all_upcoming_grilles"] = all_grilles
        if not all_grilles:
            st.warning("Aucune grille à venir trouvée.")

    all_upcoming = st.session_state.get("all_upcoming_grilles", [])
    upcoming = [g for g in all_upcoming if g["type"] == grid_type]
    if all_upcoming and not upcoming:
        st.info(f"Aucune grille {grid_type} trouvée. Types disponibles : "
                f"{', '.join(sorted(set(g['type'] for g in all_upcoming)))}")

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

        # --- Étape 2 : Charger la répartition (cotes) ---
        grid_data = st.session_state.get("grid_repartition")

        if st.button("Charger les cotes (répartition)", type="secondary"):
            with st.spinner("Chargement des cotes depuis Pronosoft..."):
                grid_data = fetch_grid_repartition(
                    grid_type,
                    selected_grille["numero"],
                )
            st.session_state["grid_repartition"] = grid_data

        # Afficher les matchs avec cotes
        if grid_data and grid_data.get("matches"):
            matches = grid_data["matches"]

            # Tableau des matchs et cotes
            rows = []
            for i, m in enumerate(matches):
                rows.append({
                    "#": i + 1,
                    "Match": f"{m.get('home', '?')} - {m.get('away', '?')}",
                    "Cote 1": f"{m.get('cote_1', 0):.2f}",
                    "Cote N": f"{m.get('cote_n', 0):.2f}",
                    "Cote 2": f"{m.get('cote_2', 0):.2f}",
                    "% 1": f"{m.get('pct_1', 0):.1f}%",
                    "% N": f"{m.get('pct_n', 0):.1f}%",
                    "% 2": f"{m.get('pct_2', 0):.1f}%",
                    "Cyborg": m.get("prono_cyborg", ""),
                })
            st.table(pd.DataFrame(rows))

            # Métriques de la grille
            if grid_data.get("difficulty"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Difficulté", f"{grid_data['difficulty']:.2f}")
                with col_b:
                    st.metric("Cote moy. favori",
                              f"{grid_data.get('moyenne_cote_fav', 0):.2f}")
                with col_c:
                    st.metric("Pronostics",
                              f"{grid_data.get('nb_pronostics', '?')}")

            # --- Recommandation de strategie ---
            try:
                from models.strategy_recommender import recommend_strategy
                grid_metrics_for_reco = {
                    "inv_spread_sum": grid_data.get("inv_spread_sum"),
                    "std_cote_fav": grid_data.get("std_cote_fav"),
                    "accord": grid_data.get("accord_cotes_joueurs"),
                    "difficulty": grid_data.get("difficulty"),
                    "moyenne_cote_fav": grid_data.get("moyenne_cote_fav"),
                }
                # Calculer les metriques manquantes
                if grid_metrics_for_reco["inv_spread_sum"] is None:
                    computed = compute_grid_metrics({"matches": matches})
                    grid_metrics_for_reco.update({
                        "inv_spread_sum": computed.get("inv_spread_sum"),
                        "std_cote_fav": computed.get("std_cote_fav"),
                        "accord": computed.get("accord_cotes_joueurs"),
                    })

                reco = recommend_strategy(grid_metrics_for_reco)
                if reco["confidence"] > 0:
                    scores_str = " | ".join(
                        f"{k}: {v:.0%}" for k, v in reco["scores"].items() if v > 0
                    )
                    st.info(
                        f"Strategie recommandee : **{reco['recommended']}** "
                        f"(confiance: {reco['confidence']:.0%})\n\n"
                        f"Scores: {scores_str}\n\n"
                        f"_{reco['reasoning']}_"
                    )
            except Exception:
                pass  # Regles pas encore generees

            # --- Analyse detaillee de la grille ---
            _show_detailed_analysis(matches, grid_type, grid_data)

        else:
            # Fallback: afficher les matchs depuis la grille à venir (sans cotes)
            for i, m in enumerate(selected_grille["matchs"]):
                st.text(f"  {i + 1}. {m['domicile']} - {m['exterieur']}")

        # --- Étape 3 : Générer les grilles ---
        # Alerte couverture cotes insuffisante
        if grid_data and grid_data.get("matches"):
            _matches = grid_data["matches"]
            _nb_sans_cotes = sum(
                1 for m in _matches
                if not (m.get("cote_1", 0) > 0 and m.get("cote_n", 0) > 0 and m.get("cote_2", 0) > 0)
            )
            if _nb_sans_cotes > 0:
                st.warning(
                    f"Attention : {_nb_sans_cotes}/{len(_matches)} matchs sans cotes. "
                    f"Le moteur utilisera des probas uniformes (1/3) pour ces matchs."
                )

        if st.button("Générer les grilles", type="primary", key="gen_grilles"):
            if not grid_data or not grid_data.get("matches"):
                st.warning("Chargez d'abord les cotes avec le bouton ci-dessus.")
            else:
                with st.spinner("Génération en cours..."):
                    # Charger les poids
                    weights = None
                    if use_optimized:
                        try:
                            from models.weight_optimizer import WeightOptimizer
                            optimizer = WeightOptimizer()
                            weights = optimizer.get_current_weights()
                            st.info(f"Poids optimisés: {weights}")
                        except Exception as e:
                            st.warning(f"Poids optimisés non disponibles: {e}")

                    predictor = OddsPredictor(
                        strategy=strategy,
                        weights=weights,
                    )
                    generator = GridGenerator(predictor=predictor, strategy=strategy)
                    grilles = generator.generate(
                        grid_data, grid_type=grid_type, budget=budget,
                    )

                if not grilles:
                    st.warning("Aucune grille générée.")
                else:
                    st.session_state["generated_grilles"] = grilles
                    st.session_state["generated_grid_number"] = selected_grille["numero"]
                    st.session_state["generated_grid_type"] = grid_type

        # Afficher les grilles generees (depuis session_state, survit aux reruns)
        gen_grilles = st.session_state.get("generated_grilles")
        if gen_grilles and grid_data and grid_data.get("matches"):
            st.success(f"{len(gen_grilles)} grille(s) générée(s)")
            for idx, grille in enumerate(gen_grilles):
                resultats = grille.get("resultats", "")
                confiance = grille.get("confiance", 0)
                probabilite = grille.get("probabilite", 0)
                profil = grille.get("profil", "")
                profil_weight = grille.get("profil_weight", 0)
                score = grille.get("score", probabilite)

                col_a, col_b, col_c, col_d = st.columns([4, 2, 2, 2])
                with col_a:
                    st.markdown(
                        f"**Grille {idx + 1}** : {format_results_html(resultats)}",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.metric("Confiance", f"{confiance:.1%}")
                with col_c:
                    st.metric("Probabilité", f"{probabilite:.4%}")
                with col_d:
                    if profil:
                        st.metric("Profil", profil)

                matchs_detail = grille.get("matchs", [])
                if matchs_detail:
                    with st.expander(f"Détails grille {idx + 1}"):
                        detail_rows = []
                        for j, md in enumerate(matchs_detail):
                            m_src = grid_data["matches"][j] if j < len(grid_data["matches"]) else {}
                            detail_rows.append({
                                "#": j + 1,
                                "Match": f"{m_src.get('home', '?')} - {m_src.get('away', '?')}",
                                "Résultat": md.get("prediction", "?"),
                                "P(1)": f"{md.get('prob_1', 0):.1%}",
                                "P(N)": f"{md.get('prob_n', 0):.1%}",
                                "P(2)": f"{md.get('prob_2', 0):.1%}",
                                "Confiance": f"{md.get('confiance', 0):.1%}",
                            })
                        st.table(pd.DataFrame(detail_rows))

                        if profil_weight:
                            st.caption(
                                f"Profil: {profil} | "
                                f"Poids profil: {profil_weight:.4f} | "
                                f"Score pondéré: {score:.6f}"
                            )

            # Bouton sauvegarder pour bankroll
            if st.button("Sauvegarder pour bankroll", key="save_bankroll"):
                st.session_state["grids_for_bankroll"] = {
                    "grilles": gen_grilles,
                    "grid_number": st.session_state.get("generated_grid_number"),
                    "grid_type": st.session_state.get("generated_grid_type", grid_type),
                }
                st.success("Grilles sauvegardees pour la bankroll. "
                           "Allez sur la page 'Suivi Bankroll' pour soumettre.")

# ---------------------------------------------------------------------------
# Page 2 — Système réducteur (doubles/triples avec garanties)
# ---------------------------------------------------------------------------

elif page == "Système réducteur":
    st.header("Système réducteur")
    st.markdown(
        "Cochez 1, 2 ou 3 résultats par match (banco / double / triple), "
        "choisissez vos garanties, et le nombre de grilles se met à jour "
        "automatiquement."
    )

    # --- Type de grille (nombre de matchs) ---
    red_type_keys = list(LOTOFOOT_TYPES.keys())
    red_type_labels = [f"{v['code']} ({k})" for k, v in LOTOFOOT_TYPES.items()]
    red_type_idx = st.selectbox(
        "Type de grille", range(len(red_type_keys)),
        format_func=lambda i: red_type_labels[i], index=0,
        key="red_grid_type",
    )
    red_grid_type = LOTOFOOT_TYPES[red_type_keys[red_type_idx]]["code"]
    red_nb_matchs = LOTOFOOT_TYPES[red_type_keys[red_type_idx]]["nb_matchs"]

    def _match_key(i: int) -> str:
        return f"red_match_{red_grid_type}_{i}"

    # --- Actions en attente (Aléatoire / Effacer), appliquées avant la
    # création des widgets pills pour pouvoir modifier leur session_state ---
    pending = st.session_state.pop("red_pending_action", None)
    if pending == "clear":
        for i in range(red_nb_matchs):
            st.session_state[_match_key(i)] = ["1"]
    elif pending == "random":
        for i in range(red_nb_matchs):
            n = random.choices([1, 2, 3], weights=[55, 32, 13])[0]
            st.session_state[_match_key(i)] = random.sample(RESULTS, n)

    # --- Sélection des doubles/triples par match ---
    st.subheader("Sélection par match")

    selections = []
    for i in range(red_nb_matchs):
        col_label, col_pills = st.columns([1, 4])
        with col_label:
            st.markdown(f"**Match {i + 1}**")
        with col_pills:
            choix = st.pills(
                f"Match {i + 1}", RESULTS, default=["1"],
                selection_mode="multi", key=_match_key(i),
                label_visibility="collapsed",
            )
            selections.append(choix or [])

    nb_matchs_incomplets = sum(1 for s in selections if not s)
    nb_doubles = sum(1 for s in selections if len(s) == 2)
    nb_triples = sum(1 for s in selections if len(s) == 3)

    nb_combinaisons = 1
    for s in selections:
        nb_combinaisons *= max(len(s), 1)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Aléatoire", key="red_random"):
            st.session_state["red_pending_action"] = "random"
            st.rerun()
    with col_btn2:
        if st.button("Effacer", key="red_clear"):
            st.session_state["red_pending_action"] = "clear"
            st.rerun()

    if nb_matchs_incomplets:
        st.warning(
            f"{nb_matchs_incomplets} match(s) sans aucun résultat sélectionné. "
            "Choisissez au moins 1 résultat par match."
        )
    st.caption(f"Doubles : {nb_doubles}  |  Triples : {nb_triples}  |  "
               f"Combinaisons possibles : {nb_combinaisons}")
    if nb_combinaisons > MAX_COMBINATIONS:
        st.error(
            f"{nb_combinaisons} combinaisons dépassent la limite du moteur "
            f"({MAX_COMBINATIONS}). Réduisez le nombre de doubles/triples."
        )

    # --- Garanties ---
    st.subheader("Garanties")

    coverage_options = ["100%", "90%", "75%", "50%"]
    coverage_map = {"100%": 1.0, "90%": 0.9, "75%": 0.75, "50%": 0.5}

    guarantees = []
    cov_n1 = st.radio(
        "Garantie N-1 (au plus 1 résultat faux)",
        coverage_options, index=0, horizontal=True, key="red_cov_n1",
    )
    guarantees.append((1, coverage_map[cov_n1]))

    if red_nb_matchs > 2:
        add_n2 = st.checkbox("Ajouter une garantie N-2 (au plus 2 résultats faux)",
                              key="red_add_n2")
        if add_n2:
            cov_n2 = st.radio(
                "Couverture N-2", coverage_options, index=1,
                horizontal=True, key="red_cov_n2", label_visibility="collapsed",
            )
            guarantees.append((2, coverage_map[cov_n2]))

            if red_nb_matchs > 3:
                add_n3 = st.checkbox(
                    "Ajouter une garantie N-3 (au plus 3 résultats faux)",
                    key="red_add_n3",
                )
                if add_n3:
                    cov_n3 = st.radio(
                        "Couverture N-3", coverage_options, index=2,
                        horizontal=True, key="red_cov_n3", label_visibility="collapsed",
                    )
                    guarantees.append((3, coverage_map[cov_n3]))

    # --- Calcul automatique du système ---
    red_result = None
    if not nb_matchs_incomplets and nb_combinaisons <= MAX_COMBINATIONS:
        try:
            with st.spinner("Calcul du système réducteur..."):
                red_result = _cached_reduction_system(
                    tuple(tuple(s) for s in selections), tuple(guarantees),
                )
        except ValueError as e:
            st.error(str(e))

    # --- Résultats ---
    st.divider()
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.metric("Nb de grilles", red_result["nb_grilles"] if red_result else "-")
    with col_g2:
        cout = f"{red_result['nb_grilles']:.2f} €" if red_result else "-"
        st.metric("Coût", cout)

    if red_result:
        methode_label = (
            "solution optimale prouvée" if red_result["method"] == "exact"
            else "solution approchée (heuristique)"
        )
        st.caption(
            f"Multiple complète : {red_result['nb_combinaisons_total']} "
            f"combinaisons — réduction de {red_result['taux_reduction']:.0%} "
            f"— {methode_label}"
        )

        couverture_rows = [
            {"Palier": f"N-{r}", "Couverture atteinte": f"{c:.0%}"}
            for r, c in sorted(red_result["couverture"].items())
        ]
        st.table(pd.DataFrame(couverture_rows))

        st.subheader("Grilles du système")
        grilles = red_result["grilles"]
        table_rows = []
        for i in range(red_nb_matchs):
            row = {"Match": i + 1}
            for gi, g in enumerate(grilles):
                row[f"Grille {gi + 1}"] = g["resultats"][i]
            table_rows.append(row)
        df_grilles = pd.DataFrame(table_rows).set_index("Match")

        def _color_result_cell(val):
            color = RESULT_COLORS.get(val, "#888")
            return f"background-color: {color}; color: white; font-weight: bold; text-align: center"

        styled_grilles = df_grilles.style.map(_color_result_cell)
        st.dataframe(styled_grilles, use_container_width=True)

        red_grid_number = st.number_input(
            "Numéro de grille", min_value=1, value=1, key="red_grid_number",
        )
        if st.button("Sauvegarder pour bankroll", key="save_bankroll_reduction"):
            st.session_state["grids_for_bankroll"] = {
                "grilles": grilles,
                "grid_number": int(red_grid_number),
                "grid_type": red_grid_type,
            }
            st.success("Grilles sauvegardées pour la bankroll. "
                       "Allez sur la page 'Suivi Bankroll' pour soumettre.")

# ---------------------------------------------------------------------------
# Page 3 — Historique des grilles (depuis JSON)
# ---------------------------------------------------------------------------

elif page == "Historique des grilles":
    st.header("Historique des grilles")

    # Sélection du type de grille
    hist_type_keys = list(LOTOFOOT_TYPES.keys())
    hist_type_labels = [f"{v['code']}" for k, v in LOTOFOOT_TYPES.items()]
    hist_type_idx = st.selectbox(
        "Type de grille", range(len(hist_type_keys)),
        format_func=lambda i: hist_type_labels[i], index=0,
        key="hist_grid_type",
    )
    hist_grid_type = LOTOFOOT_TYPES[hist_type_keys[hist_type_idx]]["code"]
    hist_nb_matchs = LOTOFOOT_TYPES[hist_type_keys[hist_type_idx]]["nb_matchs"]

    history = load_history(grid_type=hist_grid_type)

    if not history:
        st.info(f"Aucun historique {hist_grid_type} trouvé. "
                f"Lancez d'abord `python main.py scrape-history {hist_grid_type}`.")
    else:
        st.caption(f"{len(history)} grille(s) {hist_grid_type} en historique")

        data = []
        for grid in history:
            nb_m = grid.get("nb_matchs", hist_nb_matchs)
            row = {
                "N°": grid.get("grid_number", "?"),
                "Saison": grid.get("season", "-"),
                "Difficulté": grid.get("difficulty", "-"),
                "Résultats": grid.get("resultats", ""),
                "Profil": grid.get("profil", "-"),
                "Cote moy. fav": grid.get("moyenne_cote_fav", "-"),
                "Surprises": grid.get("nb_surprises", "-"),
            }
            rapports = grid.get("rapports", {})
            r_parfait = rapports.get(f"{nb_m}_sur_{nb_m}", {})
            row[f"Gagnants {nb_m}/{nb_m}"] = r_parfait.get("gagnants", "-")
            row[f"Rapport {nb_m}/{nb_m}"] = (
                f"{r_parfait.get('montant', 0):.2f} €" if r_parfait.get("montant") else "-"
            )
            data.append(row)

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Page 4 — Dashboard stats
# ---------------------------------------------------------------------------

elif page == "Dashboard stats":
    st.header("Dashboard statistiques")

    # Sélection du type pour les stats
    stats_type_keys = list(LOTOFOOT_TYPES.keys())
    stats_type_labels = [f"{v['code']}" for k, v in LOTOFOOT_TYPES.items()]
    stats_type_idx = st.selectbox(
        "Type de grille", range(len(stats_type_keys)),
        format_func=lambda i: stats_type_labels[i], index=0,
        key="stats_grid_type",
    )
    stats_grid_type = LOTOFOOT_TYPES[stats_type_keys[stats_type_idx]]["code"]
    stats_nb_matchs = LOTOFOOT_TYPES[stats_type_keys[stats_type_idx]]["nb_matchs"]

    history = load_history(grid_type=stats_grid_type)
    if not history:
        st.info("Aucun historique trouvé.")
    else:
        # Distribution des résultats
        all_results = "".join(g.get("resultats", "") for g in history)
        if all_results:
            n1 = all_results.count("1")
            nn = all_results.count("N")
            n2 = all_results.count("2")
            total = len(all_results)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("% Victoire (1)", f"{n1/total*100:.1f}%")
            with col2:
                st.metric("% Nul (N)", f"{nn/total*100:.1f}%")
            with col3:
                st.metric("% Défaite (2)", f"{n2/total*100:.1f}%")

            # Distribution
            dist_data = pd.DataFrame({
                "Résultat": ["1", "N", "2"],
                "Fréquence": [n1 / total, nn / total, n2 / total],
            })
            fig = px.bar(
                dist_data, x="Résultat", y="Fréquence",
                color="Résultat",
                color_discrete_map=RESULT_COLORS,
                text_auto=".1%",
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        # Distribution des difficultés
        difficulties = [g.get("difficulty") for g in history if g.get("difficulty")]
        if difficulties:
            st.subheader("Distribution des indices de difficulté")
            fig_diff = px.histogram(
                x=difficulties, nbins=20,
                labels={"x": "Indice de difficulté", "y": "Nombre de grilles"},
            )
            st.plotly_chart(fig_diff, use_container_width=True)

        # Distribution des surprises
        surprises = [g.get("nb_surprises", 0) for g in history]
        if surprises:
            st.subheader("Nombre de surprises par grille")
            fig_surp = px.histogram(
                x=surprises, nbins=8,
                labels={"x": "Nombre de surprises", "y": "Nombre de grilles"},
            )
            st.plotly_chart(fig_surp, use_container_width=True)

        # Stats cotes
        st.subheader("Statistiques des cotes favorites")
        cotes_data = []
        for g in history:
            if g.get("moyenne_cote_fav"):
                cotes_data.append({
                    "Grille": g.get("grid_number", "?"),
                    "Moy. cote fav": g["moyenne_cote_fav"],
                    "Écart-type": g.get("ecart_type_cotes", 0),
                    "Surprises": g.get("nb_surprises", 0),
                    "Difficulté": g.get("difficulty", 0),
                })
        if cotes_data:
            cotes_df = pd.DataFrame(cotes_data)
            fig_cotes = px.scatter(
                cotes_df, x="Moy. cote fav", y="Surprises",
                color="Difficulté",
                hover_data=["Grille"],
                labels={"Moy. cote fav": "Cote moyenne du favori"},
            )
            st.plotly_chart(fig_cotes, use_container_width=True)

        # --- Analyse des strategies ---
        st.subheader("Analyse des strategies")
        try:
            from models.strategy_recommender import build_strategy_performance_table
            perf_table = build_strategy_performance_table()
            if perf_table:
                strat_df = pd.DataFrame(perf_table)

                # Performance moyenne par strategie
                col_s1, col_s2, col_s3 = st.columns(3)
                for col_widget, strat in zip(
                    [col_s1, col_s2, col_s3],
                    ["prudente", "equilibree", "audacieuse"],
                ):
                    avg = strat_df[f"n_correct_{strat}"].mean()
                    with col_widget:
                        st.metric(f"Moy. {strat}", f"{avg:.2f}/{stats_nb_matchs}")

                # Scatter plot: inv_spread_sum vs best_strategy
                if "inv_spread_sum" in strat_df.columns:
                    scatter_df = strat_df.dropna(subset=["inv_spread_sum"])
                    if not scatter_df.empty:
                        fig_strat = px.scatter(
                            scatter_df,
                            x="inv_spread_sum",
                            y="n_correct_equilibree",
                            color="best_strategy",
                            hover_data=["grid_number", "n_correct_prudente",
                                        "n_correct_audacieuse"],
                            labels={
                                "inv_spread_sum": "Inv Spread Sum",
                                "n_correct_equilibree": "Correct (equilibree)",
                                "best_strategy": "Meilleure strategie",
                            },
                        )
                        fig_strat.update_layout(height=400)
                        st.plotly_chart(fig_strat, use_container_width=True)

                # Tableau de performance par strategie
                from collections import Counter
                best_counts = Counter(strat_df["best_strategy"])
                perf_rows = []
                for strat in ["prudente", "equilibree", "audacieuse"]:
                    perf_rows.append({
                        "Strategie": strat,
                        "Moy. correct": f"{strat_df[f'n_correct_{strat}'].mean():.2f}",
                        "Meilleure sur N grilles": best_counts.get(strat, 0),
                        "% meilleure": f"{best_counts.get(strat, 0)/len(strat_df)*100:.1f}%",
                    })
                st.table(pd.DataFrame(perf_rows))
        except Exception as e:
            st.caption(f"Analyse des strategies non disponible: {e}")

# ---------------------------------------------------------------------------
# Page 5 — Suivi Bankroll
# ---------------------------------------------------------------------------

elif page == "Suivi Bankroll":
    st.header("Suivi Bankroll")

    from bankroll.tracker import (
        load_bankroll, save_bankroll, get_current_balance,
        add_pending_entry, resolve_entry, fetch_and_resolve,
        compute_roi, get_bankroll_evolution,
    )

    bankroll_data = load_bankroll()

    # --- Metriques ---
    balance = get_current_balance(bankroll_data)
    roi = compute_roi(bankroll_data)
    entries = bankroll_data.get("entries", [])
    total_invested = sum(e.get("cost", 0) for e in entries)
    n_draws = len(entries)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Solde", f"{balance:.2f} EUR")
    with col2:
        st.metric("ROI", f"{roi:+.1f}%")
    with col3:
        st.metric("Tirages joues", n_draws)
    with col4:
        st.metric("Total investi", f"{total_invested:.2f} EUR")

    # --- Graphique evolution ---
    evolution = get_bankroll_evolution(bankroll_data)
    if len(evolution) > 1:
        st.subheader("Evolution de la bankroll")
        evo_df = pd.DataFrame(evolution)
        fig_evo = go.Figure()
        fig_evo.add_trace(go.Scatter(
            x=evo_df["date"], y=evo_df["bankroll_after"],
            mode="lines+markers", name="Bankroll",
            line=dict(color="blue", width=2),
        ))
        initial = bankroll_data.get("initial_bankroll", 100.0)
        fig_evo.add_hline(
            y=initial, line_dash="dash", line_color="gray",
            annotation_text=f"Initial ({initial:.0f} EUR)",
        )
        fig_evo.update_layout(
            xaxis_title="Date", yaxis_title="Bankroll (EUR)",
            height=350,
        )
        st.plotly_chart(fig_evo, use_container_width=True)

    # --- Soumettre des grilles ---
    st.subheader("Soumettre des grilles")

    bankroll_source = st.session_state.get("grids_for_bankroll")
    if bankroll_source:
        grilles = bankroll_source["grilles"]
        grid_number = bankroll_source["grid_number"]
        grid_type_bk = bankroll_source["grid_type"]

        st.write(f"Grilles generees pour {grid_type_bk} N°{grid_number} "
                 f"({len(grilles)} grilles)")
        for i, g in enumerate(grilles):
            st.text(f"  {i+1}. {g.get('resultats', '?')}")

        if st.button("Soumettre ces grilles", type="primary", key="submit_bankroll"):
            entry = add_pending_entry(
                bankroll_data,
                grid_number=grid_number,
                grid_type=grid_type_bk,
                submitted_grids=grilles,
            )
            save_bankroll(bankroll_data)
            st.success(f"Entree {entry['id']} ajoutee (cout: {entry['cost']:.2f} EUR)")
            del st.session_state["grids_for_bankroll"]
            st.rerun()
    else:
        st.caption("Generez des grilles sur la page 'Generateur' puis cliquez "
                   "'Sauvegarder pour bankroll'.")

    # --- Entries pending ---
    pending = [e for e in entries if e.get("status") == "pending"]
    if pending:
        st.subheader(f"Grilles en attente de resultat ({len(pending)})")
        for entry in pending:
            grids_submitted = entry.get("grids_submitted", [])
            with st.expander(
                f"{entry['id']} — {entry.get('date', '?')} — "
                f"{len(grids_submitted)} grille(s) — {entry['cost']:.2f} EUR",
                expanded=True,
            ):
                # Afficher chaque grille soumise
                for i, g in enumerate(grids_submitted):
                    resultats = g.get("resultats", "?")
                    st.markdown(
                        f"  Grille {i+1} : {format_results_html(resultats)}",
                        unsafe_allow_html=True,
                    )

                # Bouton resoudre
                if st.button("Resoudre (recuperer resultats)", key=f"resolve_{entry['id']}"):
                    with st.spinner(f"Resolution de {entry['id']}..."):
                        result = fetch_and_resolve(bankroll_data, entry["id"])
                    if result:
                        save_bankroll(bankroll_data)
                        st.success(
                            f"{entry['id']} resolu: {result['actual_results']} | "
                            f"Net: {result['net']:+.2f} EUR"
                        )
                        st.rerun()
                    else:
                        st.error(f"Echec de la resolution pour {entry['id']}.")

    # --- Historique resolu ---
    resolved = [e for e in entries if e.get("status") == "resolved"]
    if resolved:
        st.subheader("Historique des tirages")
        hist_rows = []
        for entry in resolved:
            best_correct = max(
                (g.get("n_correct", 0) for g in entry.get("grids_submitted", [])),
                default=0,
            )
            hist_rows.append({
                "ID": entry["id"],
                "Date": entry.get("date", "?"),
                "Resultats": entry.get("actual_results", ""),
                "Grilles": len(entry.get("grids_submitted", [])),
                "Meilleur": f"{best_correct}/{len(entry.get('actual_results', ''))}",
                "Cout": f"{entry['cost']:.2f}",
                "Payout": f"{entry['total_payout']:.2f}",
                "Net": f"{entry['net']:+.2f}",
                "Bankroll": f"{entry['bankroll_after']:.2f}",
            })
        st.dataframe(
            pd.DataFrame(hist_rows),
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------------------
# Page 6 — Analyse cross-types
# ---------------------------------------------------------------------------

elif page == "Analyse cross-types":
    st.header("Analyse comparative cross-types")
    st.markdown("Compare les tendances statistiques entre LF7, LF8, LF12 et LF15.")

    from analysis.cross_type_analysis import (
        load_all_types_features,
        compare_types_descriptive,
        compare_predictability,
        compare_value,
        find_cross_type_patterns,
    )

    with st.spinner("Chargement des historiques..."):
        all_features = load_all_types_features()

    if not all_features:
        st.warning("Aucun historique disponible. Lancez d'abord le scraping pour au moins un type.")
    else:
        available_types = list(all_features.keys())
        st.info(f"Types disponibles: {', '.join(f'{t} ({len(f)} grilles)' for t, f in all_features.items())}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Comparatif", "Previsibilite", "Valeur", "Patterns",
        ])

        # --- Tab 1: Tableau comparatif ---
        with tab1:
            st.subheader("Statistiques descriptives par type")
            desc = compare_types_descriptive(all_features)

            metrics_to_show = [
                ("difficulty", "Difficulte"),
                ("moy_cote_fav", "Moy. cote favori"),
                ("std_cote_fav", "Std cote favori"),
                ("moy_spread", "Moy. spread"),
                ("moy_prob_fav", "Moy. prob favori"),
                ("hhi_moyen", "HHI moyen"),
                ("gini_cotes_fav", "Gini cotes fav"),
                ("marge_bookmaker_moy", "Marge bookmaker"),
                ("nb_surprises", "Nb surprises"),
                ("pct_fav_gagne", "% fav gagne"),
                ("montant_parfait", "Montant parfait"),
                ("gagnants_parfait", "Gagnants parfait"),
                ("roi_theorique", "ROI theorique"),
                ("brier_realise", "Brier realise"),
                ("cyborg_accuracy", "Accuracy Cyborg"),
                ("joueurs_accuracy", "Accuracy joueurs"),
            ]

            table_data = []
            for metric, label in metrics_to_show:
                row = {"Metrique": label}
                for gt in available_types:
                    stats = desc["par_type"].get(gt, {}).get(metric, {})
                    row[gt] = round(stats.get("mean", 0), 4)
                table_data.append(row)

            df_comp = pd.DataFrame(table_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            # Radar chart
            st.subheader("Radar comparatif (metriques normalisees)")
            radar_metrics = [
                "moy_prob_fav", "pct_fav_gagne", "cyborg_accuracy",
                "joueurs_accuracy", "accord_cotes_joueurs", "roi_theorique",
            ]
            radar_labels = [
                "Prob favori", "% fav gagne", "Accuracy Cyborg",
                "Acc. joueurs", "Accord cotes/joueurs", "ROI theorique",
            ]

            fig_radar = go.Figure()
            for gt in available_types:
                values = []
                for metric in radar_metrics:
                    stats = desc["par_type"].get(gt, {}).get(metric, {})
                    values.append(stats.get("mean", 0))
                # Normaliser pour le radar (min-max sur les types)
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # fermer le polygone
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself",
                    name=gt,
                    opacity=0.6,
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                height=500,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- Tab 2: Previsibilite ---
        with tab2:
            st.subheader("Comparaison de la previsibilite")
            pred = compare_predictability(all_features)

            st.markdown(f"**Classement** (plus previsible en premier): "
                        f"{' > '.join(pred['classement_previsibilite'])}")

            pred_data = []
            for gt in pred["classement_previsibilite"]:
                row = {"Type": gt, "Score": pred["scores"][gt]}
                detail = pred["par_type"][gt]
                for metric in ["pct_fav_gagne", "cyborg_accuracy", "joueurs_accuracy"]:
                    row[metric] = round(detail.get(metric, {}).get("mean", 0), 4)
                pred_data.append(row)

            st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)

            # Bar chart
            fig_pred = px.bar(
                pd.DataFrame(pred_data),
                x="Type", y="Score",
                color="Type",
                title="Score de previsibilite par type",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        # --- Tab 3: Valeur ---
        with tab3:
            st.subheader("Comparaison de la valeur financiere")
            val = compare_value(all_features)

            st.markdown(f"**Classement** (meilleure valeur en premier): "
                        f"{' > '.join(val['classement_valeur'])}")

            val_data = []
            for gt in val["classement_valeur"]:
                detail = val["par_type"][gt]
                val_data.append({
                    "Type": gt,
                    "Esperance/EUR": val["esperance_par_eur"].get(gt, 0),
                    "Montant parfait (moy)": round(detail.get("montant_parfait", {}).get("mean", 0), 2),
                    "Gagnants parfait (moy)": round(detail.get("gagnants_parfait", {}).get("mean", 0), 1),
                    "ROI theorique (moy)": round(detail.get("roi_theorique", {}).get("mean", 0), 4),
                })

            st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

            fig_val = px.bar(
                pd.DataFrame(val_data),
                x="Type", y="Esperance/EUR",
                color="Type",
                title="Esperance de gain par EUR mise",
            )
            st.plotly_chart(fig_val, use_container_width=True)

        # --- Tab 4: Patterns ---
        with tab4:
            st.subheader("Patterns cross-types")

            with st.spinner("Analyse des patterns..."):
                patterns = find_cross_type_patterns(all_features)

            st.markdown("### Features les plus differenciantes entre types")
            if patterns["separating_features"]:
                sep_data = []
                for p in patterns["separating_features"][:15]:
                    row = {
                        "Metrique": p["metric"],
                        "Effect size": p["effect_size"],
                        "Paire": f"{p['best_pair'][0]} vs {p['best_pair'][1]}",
                    }
                    for gt, m in p["means"].items():
                        row[gt] = round(m, 4)
                    sep_data.append(row)
                st.dataframe(pd.DataFrame(sep_data), use_container_width=True, hide_index=True)

            st.markdown("### Correlations stables (presentes dans tous les types)")
            if patterns["stable_correlations"]:
                stable_data = []
                for c in patterns["stable_correlations"][:10]:
                    row = {"Pre-match": c["pre"], "Post-match": c["post"], "r moyen": c["r_moyen"]}
                    for gt, r in c["r_par_type"].items():
                        row[f"r {gt}"] = r
                    stable_data.append(row)
                st.dataframe(pd.DataFrame(stable_data), use_container_width=True, hide_index=True)
            else:
                st.info("Pas assez de donnees pour detecter des correlations stables.")

# ---------------------------------------------------------------------------
# Page 7 — Focus LF12
# ---------------------------------------------------------------------------

elif page == "Focus LF12":
    st.header("Focus LF12")
    st.markdown("Exploration visuelle dédiée au LF12 : gains réels par rang, tirage par tirage.")

    focus_history = load_history(grid_type="LF12")

    if not focus_history:
        st.info("Aucun historique LF12 trouvé. Lancez `python main.py scrape-history LF12`.")
    else:
        tier_labels = {"9_sur_12": "9/12", "10_sur_12": "10/12",
                        "11_sur_12": "11/12", "12_sur_12": "12/12"}
        tier_key = st.selectbox(
            "Rang à explorer", list(tier_labels.keys()),
            format_func=lambda k: tier_labels[k], index=1,  # 10/12 par défaut
        )

        rows = []
        for g in focus_history:
            tier = g.get("rapports", {}).get(tier_key)
            if not tier:
                continue
            rows.append({
                "grid_number": g.get("grid_number"),
                "season": g.get("season", "-"),
                "montant": tier.get("montant", 0) or 0,
                "gagnants": tier.get("gagnants", 0) or 0,
                "difficulty": g.get("difficulty"),
                "moyenne_cote_fav": g.get("moyenne_cote_fav"),
            })

        df_tier = pd.DataFrame(rows).sort_values("grid_number").reset_index(drop=True)
        df_gagnant = df_tier[df_tier["montant"] > 0]

        if df_gagnant.empty:
            st.warning(f"Aucun gain enregistré pour le rang {tier_labels[tier_key]}.")
        else:
            st.subheader(f"Tous les gains historiques — rang {tier_labels[tier_key]}")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Tirages payants", f"{len(df_gagnant)}/{len(df_tier)}")
            with col2:
                st.metric("Gain moyen", f"{df_gagnant['montant'].mean():.2f} €")
            with col3:
                st.metric("Gain médian", f"{df_gagnant['montant'].median():.2f} €")
            with col4:
                st.metric("Gain min", f"{df_gagnant['montant'].min():.2f} €")
            with col5:
                st.metric("Gain max", f"{df_gagnant['montant'].max():.2f} €")

            moyenne = df_gagnant["montant"].mean()

            # --- Gains dans le temps ---
            st.subheader("Gains par tirage (chronologique)")
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=df_gagnant["grid_number"], y=df_gagnant["montant"],
                marker_color="#3b82f6",
                hovertemplate="Grille n°%{x}<br>Gain: %{y:.2f} €<extra></extra>",
                name=tier_labels[tier_key],
            ))
            fig_time.add_hline(
                y=moyenne, line_dash="dash", line_color="#94a3b8",
                annotation_text=f"Moyenne ({moyenne:.2f} €)",
            )
            fig_time.update_layout(
                xaxis_title="N° de grille", yaxis_title="Gain (€)",
                height=400, showlegend=False,
                bargap=0.15,
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # --- Distribution des gains ---
            st.subheader("Distribution des gains")
            seuils = [50, 100, 200, 500]

            fig_dist = px.histogram(
                df_gagnant, x="montant", nbins=30,
                labels={"montant": "Gain (€)", "count": "Nombre de tirages"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig_dist.add_vline(
                x=moyenne, line_dash="dash", line_color="#94a3b8",
                annotation_text=f"Moyenne ({moyenne:.2f} €)",
                annotation_position="top left",
            )
            for seuil in seuils:
                pct_sous_seuil = (df_gagnant["montant"] < seuil).mean()
                fig_dist.add_vline(
                    x=seuil, line_dash="dot", line_color="#cbd5e1",
                    annotation_text=f"< {seuil}€ : {pct_sous_seuil:.0%}",
                    annotation_position="top",
                    annotation_textangle=-90,
                )
            fig_dist.update_layout(height=400, yaxis_title="Nombre de tirages")
            st.plotly_chart(fig_dist, use_container_width=True)

            # --- Repartition par tranche ---
            st.subheader("Répartition par tranche de gain")
            bornes = [0] + seuils + [float("inf")]
            tranche_labels = [
                f"< {seuils[0]}€",
                *[f"{seuils[i]}-{seuils[i+1]}€" for i in range(len(seuils) - 1)],
                f"> {seuils[-1]}€",
            ]
            tranche_idx = pd.cut(
                df_gagnant["montant"], bins=bornes, labels=tranche_labels, right=False,
            )
            tranche_counts = tranche_idx.value_counts().reindex(tranche_labels).fillna(0)
            tranche_pct = tranche_counts / tranche_counts.sum()

            fig_tranches = go.Figure(go.Bar(
                x=tranche_labels, y=tranche_pct.values,
                marker_color="#3b82f6",
                text=[f"{p:.0%}" for p in tranche_pct.values],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.1%} des tirages payants<extra></extra>",
            ))
            fig_tranches.update_layout(
                height=350, yaxis_title="Part des tirages payants",
                yaxis_tickformat=".0%", xaxis_title="Tranche de gain",
                showlegend=False,
            )
            st.plotly_chart(fig_tranches, use_container_width=True)

            # --- Lien avec une metrique pre-match (reutilisable) ---
            def _show_metric_comparison(df, metric_col, metric_label, tercile_labels):
                dfm = df.dropna(subset=[metric_col])
                if len(dfm) < 5:
                    st.info(f"Pas assez de données avec « {metric_label} » renseignée pour comparer.")
                    return

                corr = dfm[metric_col].corr(dfm["montant"])
                force = "quasi nulle" if abs(corr) < 0.2 else "faible" if abs(corr) < 0.4 else "modérée"
                st.caption(f"Corrélation {metric_label} ↔ gain : r = {corr:+.2f} ({force})")

                coeffs = np.polyfit(dfm[metric_col], dfm["montant"], 1)
                x_trend = np.linspace(dfm[metric_col].min(), dfm[metric_col].max(), 50)
                y_trend = coeffs[0] * x_trend + coeffs[1]

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=dfm[metric_col], y=dfm["montant"],
                    mode="markers", marker=dict(color="#3b82f6", size=7, opacity=0.65),
                    hovertemplate=f"{metric_label}: " + "%{x:.2f}<br>Gain: %{y:.2f} €<extra></extra>",
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=x_trend, y=y_trend, mode="lines",
                    line=dict(color="#94a3b8", dash="dash"), hoverinfo="skip",
                ))
                fig_scatter.update_layout(
                    height=400, showlegend=False,
                    xaxis_title=metric_label, yaxis_title="Gain (€)",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                dfm = dfm.copy()
                dfm["tercile"] = pd.qcut(dfm[metric_col], 3, labels=tercile_labels)
                tercile_stats = (
                    dfm.groupby("tercile", observed=True)["montant"]
                    .agg(["mean", "count"]).reset_index()
                )
                fig_tercile = go.Figure(go.Bar(
                    x=tercile_stats["tercile"], y=tercile_stats["mean"],
                    marker_color="#3b82f6",
                    text=[f"{v:.0f}€ (n={n})" for v, n in
                          zip(tercile_stats["mean"], tercile_stats["count"])],
                    textposition="outside",
                    hovertemplate="%{x}<br>Gain moyen: %{y:.2f} €<extra></extra>",
                ))
                fig_tercile.update_layout(
                    height=350, showlegend=False,
                    xaxis_title=f"{metric_label} (terciles)", yaxis_title="Gain moyen (€)",
                )
                st.plotly_chart(fig_tercile, use_container_width=True)

            st.subheader("Lien avec la difficulté de la grille")
            _show_metric_comparison(
                df_gagnant, "difficulty", "Difficulté (produit des cotes favori)",
                ["Facile", "Moyenne", "Difficile"],
            )

            st.subheader("Lien avec la moyenne des cotes favori")
            st.caption(
                "L'indicateur le plus corrélé au gain parmi ceux testés (r=+0.28). "
                "Plus la moyenne est basse, plus les favoris sont dominants sur "
                "l'ensemble de la grille (cotes proches de 1) ; plus elle est "
                "haute, plus les favoris sont fragiles match par match."
            )
            _show_metric_comparison(
                df_gagnant, "moyenne_cote_fav", "Moyenne des cotes favori",
                ["Favoris dominants", "Intermédiaire", "Favoris fragiles"],
            )

            # --- Table detaillee ---
            with st.expander("Table détaillée"):
                display_df = df_gagnant[["grid_number", "season", "montant", "gagnants"]].rename(
                    columns={"grid_number": "N° grille", "season": "Saison",
                             "montant": "Gain (€)", "gagnants": "Nb gagnants"}
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)
