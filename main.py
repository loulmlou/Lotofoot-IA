"""Point d'entrée principal de LotoFoot AI Analyst."""

import sys
import json

from loguru import logger


def cmd_scrape_history():
    """Scrape l'historique des grilles depuis Pronosoft.

    Usage: python main.py scrape-history [start] [end] [grid_type]
    """
    from collectors.pronosoft_history import fetch_all_history, save_history

    # Parse args: scrape-history [start] [end] [type]
    # Or: scrape-history [type] (if arg is not a number)
    args = sys.argv[2:]
    grid_type = "LF7"
    start, end = 1, 91

    # Extract grid_type if any arg looks like LF*
    for a in args:
        if a.upper().startswith("LF"):
            grid_type = a.upper()
            args = [x for x in args if x != a]
            break

    if len(args) >= 1:
        start = int(args[0])
    if len(args) >= 2:
        end = int(args[1])

    logger.info(f"Scraping des grilles {grid_type} de {start} à {end}...")
    grids = fetch_all_history(start=start, end=end, grid_type=grid_type, delay=1.0)
    save_history(grids, grid_type=grid_type)
    logger.info(f"Terminé: {len(grids)} grilles sauvegardées.")


def cmd_scrape_multi_season():
    """Scrape l'historique sur plusieurs saisons.

    Usage: python main.py scrape-multi-season [n_seasons] [grid_type]
    """
    from collectors.pronosoft_history import (
        fetch_multi_season_history, save_history, load_history,
        AVAILABLE_SEASONS,
    )

    # Parse args
    args = sys.argv[2:]
    grid_type = "LF7"
    n_seasons_arg = None

    for a in args:
        if a.upper().startswith("LF"):
            grid_type = a.upper()
        else:
            try:
                n_seasons_arg = int(a)
            except ValueError:
                pass

    available = AVAILABLE_SEASONS.get(grid_type, AVAILABLE_SEASONS["LF7"])
    n_seasons = n_seasons_arg if n_seasons_arg is not None else len(available)
    seasons = available[:n_seasons]

    print(f"Scraping {grid_type} — {n_seasons} saison(s):")
    for s, y, n in seasons:
        print(f"  {s} (year={y}, ~{n} grilles)")

    grids = fetch_multi_season_history(seasons=seasons, grid_type=grid_type, delay=1.5)

    # Fusionner avec l'historique existant (eviter les doublons)
    existing = load_history(grid_type=grid_type)
    existing_keys = {(g.get("season", f"{g['year']-1}-{g['year']}"), g["grid_number"])
                     for g in existing}

    new_count = 0
    for g in grids:
        key = (g.get("season", f"{g['year']-1}-{g['year']}"), g["grid_number"])
        if key not in existing_keys:
            existing.append(g)
            existing_keys.add(key)
            new_count += 1

    # Trier par saison puis numero
    existing.sort(key=lambda g: (g.get("season", ""), g.get("grid_number", 0)))

    save_history(existing, grid_type=grid_type)
    print(f"\nTermine: {new_count} nouvelles grilles {grid_type} ajoutees, "
          f"{len(existing)} grilles au total.")


def cmd_optimize_weights():
    """Optimise les poids du modèle sur l'historique."""
    from models.weight_optimizer import WeightOptimizer

    metric = sys.argv[2] if len(sys.argv) > 2 else "accuracy"
    logger.info(f"Optimisation des poids (métrique: {metric})...")

    optimizer = WeightOptimizer()
    result = optimizer.optimize_global(metric=metric, n_restarts=10)

    print(f"\nMeilleurs poids ({metric}):")
    print(json.dumps(result["weights"], indent=2))
    print(f"Score: {result['score']:.4f}")

    # Afficher aussi les poids courants (fenêtre glissante)
    current = optimizer.get_current_weights()
    print(f"\nPoids courants (dernières 20 grilles):")
    print(json.dumps(current, indent=2))


def cmd_backtest():
    """Lance le backtesting complet.

    Usage: python main.py backtest [grid_type]
    """
    from backtesting.engine import run_full_backtest

    grid_type = "LF7"
    for a in sys.argv[2:]:
        if a.upper().startswith("LF"):
            grid_type = a.upper()

    run_full_backtest(grid_type=grid_type)


def cmd_analyze_strategies():
    """Analyse les performances des 3 strategies sur l'historique.

    Usage: python main.py analyze-strategies [grid_type]
    """
    from models.strategy_recommender import (
        build_strategy_performance_table,
        build_recommendation_rules,
    )

    grid_type = "LF7"
    for a in sys.argv[2:]:
        if a.upper().startswith("LF"):
            grid_type = a.upper()

    print(f"\n=== Analyse des strategies ({grid_type}) ===\n")
    table = build_strategy_performance_table(grid_type=grid_type)
    if not table:
        print("Aucune grille analysee.")
        return

    # Determine nb_matchs
    nb_matchs = 7
    if table:
        nb_matchs = table[0].get("nb_matchs", 7)

    # Afficher la table
    print(f"{'Grille':>7}  {'Prud':>4}  {'Equi':>4}  {'Auda':>4}  {'Best':>12}  {'Marge':>5}")
    print("-" * 50)
    for row in table:
        print(
            f"{row['grid_number']:>7}  "
            f"{row['n_correct_prudente']:>4}  "
            f"{row['n_correct_equilibree']:>4}  "
            f"{row['n_correct_audacieuse']:>4}  "
            f"{row['best_strategy']:>12}  "
            f"{row['margin']:>5}"
        )

    # Stats globales
    from collections import Counter
    best_counts = Counter(r["best_strategy"] for r in table)
    print(f"\nMeilleure strategie par grille:")
    for name, count in best_counts.most_common():
        print(f"  {name}: {count} ({count/len(table):.1%})")

    avg_scores = {}
    for strat in ["prudente", "equilibree", "audacieuse"]:
        avg = sum(r[f"n_correct_{strat}"] for r in table) / len(table)
        avg_scores[strat] = avg
        print(f"  Moy. correct {strat}: {avg:.2f}/{nb_matchs}")

    # Construire les regles
    print("\n--- Construction des regles ---")
    rules_data = build_recommendation_rules(table)
    print(f"Accuracy globale des regles: {rules_data['accuracy']:.1%}")
    for rule in rules_data.get("rules", []):
        direction = "<" if rule["direction"] == "low" else ">="
        print(
            f"  {rule['metric']} {direction} {rule['threshold']:.2f} "
            f"-> {rule['recommended']} (acc={rule['accuracy']:.0%}, n={rule['support']})"
        )


def _print_utf8(text: str):
    """Print UTF-8 text on Windows without encoding errors."""
    import sys as _sys
    _sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    _sys.stdout.buffer.write(b"\n")
    _sys.stdout.buffer.flush()


def cmd_correlation_study():
    """Lance l'étude de corrélations pour un type de grille.

    Usage: python main.py correlation-study [LF7|LF8|LF12|LF15]
    """
    from analysis.correlation_study import generate_report

    grid_type = "LF7"
    for a in sys.argv[2:]:
        if a.upper().startswith("LF"):
            grid_type = a.upper()

    print(f"\nAnalyse des correlations pour {grid_type}...")
    report = generate_report(grid_type=grid_type)
    _print_utf8(report)

    # Sauvegarder
    import os
    report_path = os.path.join("data", "history", f"correlation_report_{grid_type}.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nRapport sauvegarde: {report_path}")


def cmd_cross_analysis():
    """Lance l'analyse comparative cross-types.

    Usage: python main.py cross-analysis
    """
    from analysis.cross_type_analysis import generate_cross_type_report

    print("\nAnalyse comparative cross-types en cours...")
    report = generate_cross_type_report()
    _print_utf8(report)

    # Sauvegarder
    import os
    report_path = os.path.join("data", "history", "cross_type_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nRapport sauvegarde: {report_path}")


def cmd_resolve_bankroll():
    """Resout une entree bankroll pending."""
    from bankroll.tracker import load_bankroll, save_bankroll, fetch_and_resolve

    entry_id = sys.argv[2] if len(sys.argv) > 2 else None
    if not entry_id:
        print("Usage: python main.py resolve-bankroll <entry_id>")
        print("  Exemple: python main.py resolve-bankroll LF7-92")
        return

    data = load_bankroll()
    result = fetch_and_resolve(data, entry_id)
    if result:
        save_bankroll(data)
        print(f"\nEntree {entry_id} resolue:")
        print(f"  Resultats reels: {result['actual_results']}")
        nb_m = len(result.get("actual_results", ""))
        for g in result["grids_submitted"]:
            print(f"  Grille {g['resultats']}: {g['n_correct']}/{nb_m} -> {g['payout']:.2f} EUR")
        print(f"  Total payout: {result['total_payout']:.2f} EUR")
        print(f"  Net: {result['net']:+.2f} EUR")
        print(f"  Bankroll: {result['bankroll_after']:.2f} EUR")
    else:
        print(f"Echec de la resolution pour {entry_id}.")


def cmd_predict():
    """Prédit la grille LF7 courante."""
    from collectors.pronosoft_scraper import (
        fetch_upcoming_grilles_pronosoft,
        fetch_grid_repartition,
    )
    from models.odds_predictor import OddsPredictor
    from generator.grid_generator import GridGenerator

    logger.info("Recherche de la grille LF7 à venir...")
    grilles = fetch_upcoming_grilles_pronosoft()
    lf7_grilles = [g for g in grilles if g["type"] == "LF7"]

    if not lf7_grilles:
        logger.error("Aucune grille LF7 à venir trouvée.")
        return

    grille = lf7_grilles[0]
    logger.info(f"Grille LF7 N°{grille['numero']} trouvée ({len(grille['matchs'])} matchs)")

    # Charger la répartition
    logger.info("Chargement des cotes...")
    grid_data = fetch_grid_repartition("LF7", grille["numero"])

    if not grid_data.get("matches"):
        logger.warning("Cotes non disponibles, utilisation des données de base.")
        # Fallback: construire grid_data minimal
        grid_data = {
            "matches": [
                {"home": m["domicile"], "away": m["exterieur"],
                 "cote_1": 0, "cote_n": 0, "cote_2": 0}
                for m in grille["matchs"]
            ]
        }

    # Prédiction
    strategy = sys.argv[2] if len(sys.argv) > 2 else "equilibree"
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    # Essayer d'utiliser les poids optimisés
    weights = None
    try:
        from models.weight_optimizer import WeightOptimizer
        optimizer = WeightOptimizer()
        if optimizer.grids:
            weights = optimizer.get_current_weights()
            logger.info(f"Poids optimisés chargés: {weights}")
    except Exception:
        pass

    predictor = OddsPredictor(strategy=strategy, weights=weights)
    generator = GridGenerator(predictor=predictor, strategy=strategy)
    grilles_gen = generator.generate(grid_data, grid_type="LF7", budget=budget)

    if not grilles_gen:
        logger.warning("Aucune grille générée.")
        return

    print(f"\n{'='*60}")
    print(f"PRÉDICTIONS LF7 N°{grille['numero']} — Stratégie: {strategy}")
    print(f"{'='*60}")

    for i, g in enumerate(grilles_gen):
        resultats = g["resultats"]
        prob = g["probabilite"]
        conf = g["confiance"]
        profil = g.get("profil", "")
        score = g.get("score", prob)

        print(f"\nGrille {i+1}: {resultats}  "
              f"(prob={prob:.4%}, conf={conf:.1%}, profil={profil})")

        for j, m in enumerate(g["matchs"]):
            src = grid_data["matches"][j] if j < len(grid_data["matches"]) else {}
            home = src.get("home", "?")
            away = src.get("away", "?")
            pred = m["prediction"]
            print(f"  {j+1}. {home:20s} - {away:20s}  →  {pred}  "
                  f"(1={m['prob_1']:.1%}  N={m['prob_n']:.1%}  2={m['prob_2']:.1%})")


def main():
    """Point d'entrée par défaut."""
    logger.info("=== LotoFoot AI Analyst ===")
    print("\nCommandes disponibles:")
    print("  python main.py scrape-history [start] [end] [LF7|LF8|LF12|LF15]")
    print("  python main.py scrape-multi-season [n] [LF7|LF8|LF12|LF15]")
    print("  python main.py optimize-weights [metric]")
    print("  python main.py backtest [LF7|LF8|LF12|LF15]")
    print("  python main.py predict [strategy] [budget]")
    print("  python main.py analyze-strategies [LF7|LF8|LF12|LF15]")
    print("  python main.py correlation-study [LF7|LF8|LF12|LF15]")
    print("  python main.py cross-analysis")
    print("  python main.py resolve-bankroll <entry_id>")
    print("  python main.py ui")
    print("  python main.py serve")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "serve":
            import uvicorn
            uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
        elif cmd == "ui":
            import subprocess
            subprocess.run(["streamlit", "run", "frontend/app.py"])
        elif cmd == "scrape-history":
            cmd_scrape_history()
        elif cmd == "scrape-multi-season":
            cmd_scrape_multi_season()
        elif cmd == "optimize-weights":
            cmd_optimize_weights()
        elif cmd == "backtest":
            cmd_backtest()
        elif cmd == "predict":
            cmd_predict()
        elif cmd == "analyze-strategies":
            cmd_analyze_strategies()
        elif cmd == "correlation-study":
            cmd_correlation_study()
        elif cmd == "cross-analysis":
            cmd_cross_analysis()
        elif cmd == "resolve-bankroll":
            cmd_resolve_bankroll()
        else:
            print(f"Commande inconnue: {cmd}")
            main()
    else:
        main()
