"""Suivi de bankroll pour LotoFoot AI.

Gere la persistance des mises, le calcul des gains/pertes,
et l'evolution de la bankroll dans le temps.
"""

import json
import os
from datetime import date

from loguru import logger


DEFAULT_BANKROLL_PATH = os.path.join("data", "bankroll", "bankroll.json")

DEFAULT_BANKROLL = {
    "initial_bankroll": 100.0,
    "cost_per_grid": 1.0,
    "grids_per_draw": 5,
    "entries": [],
}


def load_bankroll(path: str = DEFAULT_BANKROLL_PATH) -> dict:
    """Charge les donnees de bankroll depuis le fichier JSON.

    Returns:
        dict avec initial_bankroll, cost_per_grid, entries, etc.
    """
    if not os.path.exists(path):
        logger.info(f"Fichier bankroll non trouve, creation: {path}")
        return DEFAULT_BANKROLL.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Erreur lecture bankroll: {e}")
        return DEFAULT_BANKROLL.copy()


def save_bankroll(data: dict, path: str = DEFAULT_BANKROLL_PATH) -> None:
    """Sauvegarde les donnees de bankroll en JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Bankroll sauvegardee: {path}")


def get_current_balance(data: dict) -> float:
    """Retourne le solde courant de la bankroll."""
    entries = data.get("entries", [])
    if not entries:
        return data.get("initial_bankroll", 100.0)
    return entries[-1].get("bankroll_after", data.get("initial_bankroll", 100.0))


def add_pending_entry(data: dict, grid_number: int, grid_type: str,
                      submitted_grids: list[dict], entry_date: str = None) -> dict:
    """Ajoute une entree pending (mises soumises, pas encore resolues).

    Args:
        data: donnees bankroll
        grid_number: numero de la grille
        grid_type: type (LF7, LF8, etc.)
        submitted_grids: liste de dicts avec 'resultats' (ex: '1N21121')
        entry_date: date au format YYYY-MM-DD (defaut: aujourd'hui)

    Returns:
        l'entree creee
    """
    cost_per_grid = data.get("cost_per_grid", 1.0)
    cost = len(submitted_grids) * cost_per_grid
    balance = get_current_balance(data)

    entry_id = f"{grid_type}-{grid_number}"
    if entry_date is None:
        entry_date = date.today().isoformat()

    grids_submitted = []
    for g in submitted_grids:
        grids_submitted.append({
            "resultats": g.get("resultats", ""),
            "n_correct": None,
            "payout": 0.0,
        })

    entry = {
        "id": entry_id,
        "grid_number": grid_number,
        "grid_type": grid_type,
        "date": entry_date,
        "status": "pending",
        "grids_submitted": grids_submitted,
        "cost": cost,
        "total_payout": 0.0,
        "net": -cost,
        "bankroll_after": balance - cost,
        "actual_results": "",
        "rapports": {},
    }

    data.setdefault("entries", []).append(entry)
    return entry


def resolve_entry(data: dict, entry_id: str,
                  actual_results: str, rapports: dict) -> dict | None:
    """Resout une entree pending en comparant les grilles soumises au resultat reel.

    Args:
        data: donnees bankroll
        entry_id: identifiant de l'entree (ex: 'LF7-92')
        actual_results: resultats reels (ex: '1N21121')
        rapports: dict des rapports {rang_key: {gagnants, montant}}

    Returns:
        l'entree mise a jour, ou None si non trouvee
    """
    entry = None
    entry_idx = None
    for i, e in enumerate(data.get("entries", [])):
        if e["id"] == entry_id:
            entry = e
            entry_idx = i
            break

    if entry is None:
        logger.warning(f"Entree {entry_id} non trouvee.")
        return None

    entry["actual_results"] = actual_results
    entry["rapports"] = rapports
    entry["status"] = "resolved"

    total_payout = 0.0
    for grid in entry["grids_submitted"]:
        predicted = grid["resultats"]
        n_correct = sum(
            1 for a, b in zip(predicted, actual_results)
            if a == b
        )
        grid["n_correct"] = n_correct

        # Lookup payout dans les rapports
        nb_matchs = len(actual_results)
        rang_key = f"{n_correct}_sur_{nb_matchs}"
        rapport = rapports.get(rang_key, {})
        payout = rapport.get("montant", 0.0)
        grid["payout"] = payout
        total_payout += payout

    entry["total_payout"] = total_payout
    entry["net"] = total_payout - entry["cost"]

    # Recalculer bankroll_after pour cette entree et les suivantes
    _recalculate_balances(data, entry_idx)

    return entry


def fetch_and_resolve(data: dict, entry_id: str) -> dict | None:
    """Auto-fetch resultats depuis Pronosoft et resout l'entree.

    Args:
        data: donnees bankroll
        entry_id: identifiant de l'entree

    Returns:
        l'entree mise a jour, ou None si echec
    """
    entry = None
    for e in data.get("entries", []):
        if e["id"] == entry_id:
            entry = e
            break

    if entry is None:
        logger.warning(f"Entree {entry_id} non trouvee.")
        return None

    grid_number = entry["grid_number"]
    grid_type = entry.get("grid_type", "LF7")

    from collectors.pronosoft_history import fetch_grid_history
    grid_data = fetch_grid_history(grid_number, grid_type=grid_type)
    if not grid_data:
        logger.error(f"Impossible de recuperer les donnees pour la grille {grid_number}.")
        return None

    actual_results = grid_data.get("resultats", "")
    rapports = grid_data.get("rapports", {})

    if not actual_results:
        logger.warning(f"Resultats non disponibles pour la grille {grid_number}.")
        return None

    # Verifier que le nombre de resultats correspond au type de grille
    expected_nb = {"LF7": 7, "LF8": 8, "LF12": 12, "LF15": 15}.get(grid_type, 7)
    if len(actual_results) < expected_nb:
        logger.warning(
            f"Resultats incomplets pour {grid_type} n{grid_number}: "
            f"{len(actual_results)}/{expected_nb} matchs joues. "
            f"Grille pas encore terminee."
        )
        return None

    return resolve_entry(data, entry_id, actual_results, rapports)


def compute_roi(data: dict) -> float:
    """Calcule le ROI en pourcentage.

    ROI = (balance_actuelle - initial) / total_investi * 100
    """
    initial = data.get("initial_bankroll", 100.0)
    balance = get_current_balance(data)
    total_invested = sum(e.get("cost", 0) for e in data.get("entries", []))
    if total_invested == 0:
        return 0.0
    return (balance - initial) / total_invested * 100


def get_bankroll_evolution(data: dict) -> list[dict]:
    """Retourne l'evolution de la bankroll pour graphique.

    Returns:
        Liste de dicts {date, bankroll_after, net}
    """
    initial = data.get("initial_bankroll", 100.0)
    evolution = [{"date": "Initial", "bankroll_after": initial, "net": 0.0}]

    for entry in data.get("entries", []):
        evolution.append({
            "date": entry.get("date", "?"),
            "bankroll_after": entry.get("bankroll_after", 0),
            "net": entry.get("net", 0),
        })

    return evolution


def _recalculate_balances(data: dict, from_idx: int = 0) -> None:
    """Recalcule bankroll_after pour toutes les entrees a partir de from_idx."""
    entries = data.get("entries", [])
    if not entries:
        return

    initial = data.get("initial_bankroll", 100.0)

    for i in range(len(entries)):
        if i == 0:
            prev_balance = initial
        else:
            prev_balance = entries[i - 1]["bankroll_after"]

        if i >= from_idx:
            entries[i]["bankroll_after"] = prev_balance - entries[i]["cost"] + entries[i]["total_payout"]
