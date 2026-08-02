"""Systèmes réducteurs (grilles réduites) à garanties pour Loto Foot.

Étant donné une sélection de doubles/triples par match, calcule un
sous-ensemble minimal de grilles (parmi la multiple complète) qui
garantit, pour chaque palier demandé (ex: n-1 à 100%, n-2 à 75%),
qu'au moins ce pourcentage des combinaisons possibles aura une grille
jouée à au plus r résultats faux.

C'est un problème de couverture en distance de Hamming, résolu ici
par un algorithme glouton (approximation standard du set-cover).
"""

from itertools import product

import numpy as np
from loguru import logger


RESULTS = ["1", "N", "2"]

# Taille max de la multiple complète (nb de combinaisons) au-delà de
# laquelle le calcul des distances par paires devient trop coûteux.
MAX_COMBINATIONS = 5000


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
        }

    # Encodage numérique pour calcul rapide des distances de Hamming
    result_idx = {r: i for i, r in enumerate(RESULTS)}
    matrix = np.array([[result_idx[c] for c in combo] for combo in combos])

    # Matrice des distances de Hamming entre toutes les paires de combinaisons
    dist_matrix = np.sum(matrix[:, None, :] != matrix[None, :, :], axis=2)

    selected_mask = np.zeros(total, dtype=bool)
    selected_indices: list = []
    couverture_reelle = {}

    for r, coverage_cible in guarantees:
        nb_cible = int(np.ceil(coverage_cible * total))
        within_r = dist_matrix <= r  # (total, total)

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

    grilles = [{"resultats": "".join(combos[i])} for i in selected_indices]

    logger.info(
        f"Système réducteur: {len(grilles)} grilles pour {total} combinaisons "
        f"(réduction {100 * (1 - len(grilles) / total):.1f}%)"
    )

    return {
        "grilles": grilles,
        "nb_grilles": len(grilles),
        "nb_combinaisons_total": total,
        "taux_reduction": 1 - len(grilles) / total,
        "couverture": couverture_reelle,
    }
