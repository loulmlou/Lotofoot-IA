"""Systèmes réducteurs (grilles réduites) à garanties pour Loto Foot.

Étant donné une sélection de doubles/triples par match, calcule un
sous-ensemble minimal de grilles (parmi la multiple complète) qui
garantit, pour chaque palier demandé (ex: n-1 à 100%, n-2 à 75%),
qu'au moins ce pourcentage des combinaisons possibles aura une grille
jouée à au plus r résultats faux.

C'est un problème de couverture en distance de Hamming (le "football
pool problem" classique en combinatoire). Deux méthodes de résolution :

- exacte : programme linéaire en nombres entiers (minimum set cover
  multi-paliers), garanti optimal, utilisé sous EXACT_SOLVER_MAX_COMBINATIONS.
- gloutonne : approximation rapide (set-cover greedy), utilisée en
  repli au-delà de cette limite ou si le solveur exact échoue à temps.
"""

from itertools import product

import numpy as np
from loguru import logger
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


RESULTS = ["1", "N", "2"]

# Taille max de la multiple complète (nb de combinaisons) au-delà de
# laquelle le calcul des distances par paires devient trop coûteux.
MAX_COMBINATIONS = 5000

# En dessous de ce nombre de combinaisons, on tente d'abord un solveur
# exact (garanti optimal). Au-delà, le programme en nombres entiers
# devient trop lent à résoudre et on retombe sur l'heuristique gloutonne.
EXACT_SOLVER_MAX_COMBINATIONS = 300
EXACT_SOLVER_TIME_LIMIT = 20  # secondes


def _validate_selections(selections: list) -> None:
    if not selections:
        raise ValueError("selections ne peut pas être vide")
    for i, options in enumerate(selections):
        if not options:
            raise ValueError(f"match {i}: aucune option sélectionnée")
        if len(set(options)) != len(options):
            raise ValueError(f"match {i}: options en double {options}")
        if not set(options) <= set(RESULTS):
            raise ValueError(
                f"match {i}: options invalides {options} (attendu parmi {RESULTS})"
            )


def _validate_guarantees(guarantees: list, n_matchs: int) -> list:
    if not guarantees:
        raise ValueError("guarantees ne peut pas être vide")
    for r, coverage in guarantees:
        if not (0 <= r < n_matchs):
            raise ValueError(
                f"rayon de garantie invalide: {r} (doit être entre 0 et {n_matchs - 1})"
            )
        if not (0 < coverage <= 1.0):
            raise ValueError(f"couverture invalide: {coverage} (doit être dans ]0, 1])")
    return sorted(guarantees, key=lambda g: g[0])


def _solve_greedy(dist_matrix: np.ndarray, guarantees: list, total: int,
                   n_matchs: int) -> tuple:
    """Heuristique gloutonne : ajoute à chaque étape la grille couvrant le
    plus de combinaisons non encore garanties. Rapide mais pas garantie
    optimale (peut nécessiter une grille de plus que le minimum absolu).
    """
    selected_mask = np.zeros(total, dtype=bool)
    selected_indices: list = []
    couverture_reelle = {}

    for r, coverage_cible in guarantees:
        nb_cible = int(np.ceil(coverage_cible * total))
        within_r = dist_matrix <= r

        if selected_indices:
            dists = dist_matrix[selected_indices, :].min(axis=0)
        else:
            dists = np.full(total, n_matchs + 1)

        couverts = dists <= r
        nb_couverts = int(np.sum(couverts))

        while nb_couverts < nb_cible:
            non_couverts = ~couverts
            gains = np.sum(within_r & non_couverts[None, :], axis=1)
            gains[selected_mask] = -1

            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                break  # ne devrait pas arriver si coverage_cible <= 1.0

            selected_indices.append(best_idx)
            selected_mask[best_idx] = True

            dists = np.minimum(dists, dist_matrix[best_idx])
            couverts = dists <= r
            nb_couverts = int(np.sum(couverts))

        couverture_reelle[r] = nb_couverts / total

    return selected_indices, couverture_reelle


def _solve_exact(dist_matrix: np.ndarray, guarantees: list, total: int,
                  time_limit: float = EXACT_SOLVER_TIME_LIMIT) -> tuple | None:
    """Résout le système réducteur de façon exacte via un programme
    linéaire en nombres entiers (minimum set cover multi-paliers).

    Variables : x_j (grille j sélectionnée ou non) et, pour chaque palier
    t, y_{t,i} (combinaison i comptée comme couverte au palier t). On
    minimise le nombre de grilles sous contrainte que chaque palier
    atteigne sa couverture cible.

    Retourne (selected_indices, couverture) si une solution optimale est
    prouvée dans le temps imparti, sinon None (repli sur le glouton).
    """
    n_tiers = len(guarantees)
    n_vars = total + total * n_tiers

    c = np.zeros(n_vars)
    c[:total] = 1.0

    constraints = []
    for t_idx, (r, coverage) in enumerate(guarantees):
        within_r = (dist_matrix <= r).astype(float)
        y_offset = total + t_idx * total

        # y_{t,i} <= somme_j within_r[i,j] * x_j
        A_cov = lil_matrix((total, n_vars))
        A_cov[:, :total] = within_r
        A_cov[np.arange(total), y_offset + np.arange(total)] = -1
        constraints.append(LinearConstraint(A_cov.tocsr(), lb=0, ub=np.inf))

        # somme_i y_{t,i} >= nb_cible
        nb_cible = int(np.ceil(coverage * total))
        A_tier = lil_matrix((1, n_vars))
        A_tier[0, y_offset:y_offset + total] = 1
        constraints.append(LinearConstraint(A_tier.tocsr(), lb=nb_cible, ub=np.inf))

    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)

    result = milp(
        c, constraints=constraints, integrality=integrality, bounds=bounds,
        options={"time_limit": time_limit, "disp": False},
    )

    if result.status != 0 or result.x is None:
        return None

    selected_indices = [i for i in range(total) if result.x[i] > 0.5]

    couverture_reelle = {}
    for r, _ in guarantees:
        if selected_indices:
            couverts = int(np.sum(np.any(dist_matrix[selected_indices, :] <= r, axis=0)))
        else:
            couverts = 0
        couverture_reelle[r] = couverts / total

    return selected_indices, couverture_reelle


def build_reduction_system(selections: list, guarantees: list) -> dict:
    """Construit un système réducteur satisfaisant les paliers de garantie.

    Args:
        selections: liste de listes de résultats possibles par match,
                    ex: [["1"], ["1", "N"], ["2"], ["1", "N", "2"], ...]
                    (1 option = banco/simple, 2 = double, 3 = triple)
        guarantees: liste de tuples (r, couverture) où r = nombre de
                    résultats faux toléré (n-r bons garantis) et
                    couverture = fraction (0-1] des combinaisons possibles
                    devant être couvertes à ce rayon.
                    Ex: [(1, 1.0), (2, 0.75)] = garantie n-1 100%, n-2 75%.

    Returns:
        dict {
            grilles: list[dict] ({resultats: str}), grilles du système,
            nb_grilles: int,
            nb_combinaisons_total: int (taille de la multiple complète),
            taux_reduction: float (1 - nb_grilles / nb_combinaisons_total),
            couverture: dict {r: fraction réellement couverte},
            method: "exact" (optimal prouvé) ou "greedy" (approximation),
        }
    """
    _validate_selections(selections)
    n_matchs = len(selections)
    guarantees = _validate_guarantees(guarantees, n_matchs)

    combos = list(product(*selections))
    total = len(combos)

    if total > MAX_COMBINATIONS:
        raise ValueError(
            f"Trop de combinaisons ({total}) pour un système réducteur "
            f"(limite {MAX_COMBINATIONS}). Réduisez le nombre de doubles/triples."
        )

    logger.info(f"Système réducteur: {n_matchs} matchs, {total} combinaisons possibles")

    if total == 1:
        return {
            "grilles": [{"resultats": "".join(combos[0])}],
            "nb_grilles": 1,
            "nb_combinaisons_total": 1,
            "taux_reduction": 0.0,
            "couverture": {r: 1.0 for r, _ in guarantees},
            "method": "exact",
        }

    # Encodage numérique pour calcul rapide des distances de Hamming
    result_idx = {r: i for i, r in enumerate(RESULTS)}
    matrix = np.array([[result_idx[c] for c in combo] for combo in combos])

    # Matrice des distances de Hamming entre toutes les paires de combinaisons
    dist_matrix = np.sum(matrix[:, None, :] != matrix[None, :, :], axis=2)

    method = "greedy"
    selected_indices = None
    couverture_reelle = None

    if total <= EXACT_SOLVER_MAX_COMBINATIONS:
        exact = _solve_exact(dist_matrix, guarantees, total)
        if exact is not None:
            selected_indices, couverture_reelle = exact
            method = "exact"
        else:
            logger.warning(
                "Solveur exact n'a pas trouvé de solution optimale dans le temps "
                "imparti, repli sur l'heuristique gloutonne."
            )

    if selected_indices is None:
        selected_indices, couverture_reelle = _solve_greedy(
            dist_matrix, guarantees, total, n_matchs,
        )

    grilles = [{"resultats": "".join(combos[i])} for i in selected_indices]

    logger.info(
        f"Système réducteur ({method}): {len(grilles)} grilles pour {total} "
        f"combinaisons (réduction {100 * (1 - len(grilles) / total):.1f}%)"
    )

    return {
        "grilles": grilles,
        "nb_grilles": len(grilles),
        "nb_combinaisons_total": total,
        "taux_reduction": 1 - len(grilles) / total,
        "couverture": couverture_reelle,
        "method": method,
    }
