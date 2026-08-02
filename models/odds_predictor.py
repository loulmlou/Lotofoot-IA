"""Prédicteur basé sur les cotes bookmakers (Pronosoft).

Remplace models/predictor.py. Produit le même format de sortie
(prob_1, prob_n, prob_2, prediction, confiance, probas) pour rester
compatible avec generator/optimizer.py.
"""

import numpy as np
from loguru import logger


# Distribution historique 1N2 connue (Loto Foot)
# Reequilibree pour reduire le biais domicile excessif
# Historique brut etait [0.43, 0.27, 0.30] mais les grilles LotoFoot
# incluent des matchs internationaux/neutres ou le biais dom est moindre.
HISTORICAL_1N2 = np.array([0.40, 0.29, 0.31])

# Poids par défaut des signaux
DEFAULT_WEIGHTS = {
    "cotes": 0.60,
    "consensus": 0.15,
    "cyborg": 0.10,
    "difficulte": 0.10,
    "historique": 0.05,
}

# Seuils de difficulté pour la sélection de stratégie
DEFAULT_DIFFICULTY_THRESHOLDS = {
    "secure_max_difficulty": 5.0,
    "secure_max_cote_fav": 1.70,
    "risky_min_difficulty": 8.0,
    "risky_min_cote_fav": 2.20,
}


class OddsPredictor:
    """Prédicteur basé sur les cotes bookmakers et signaux Pronosoft."""

    def __init__(self, strategy: str = "equilibree", weights: dict = None,
                 difficulty_thresholds: dict = None):
        """Initialise le prédicteur.

        Args:
            strategy: stratégie par défaut ('prudente', 'equilibree', 'audacieuse')
            weights: poids des signaux (cotes, consensus, cyborg, difficulte, historique)
            difficulty_thresholds: seuils pour la sélection de profil de stratégie
        """
        self.strategy = strategy
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.difficulty_thresholds = (
            difficulty_thresholds or DEFAULT_DIFFICULTY_THRESHOLDS.copy()
        )

    def predict_match(self, match_data: dict,
                      grid_metrics: dict = None) -> dict:
        """Prédit le résultat d'un match à partir des cotes et signaux.

        Args:
            match_data: dict avec cote_1, cote_n, cote_2, pct_1, pct_n, pct_2,
                        prono_cyborg (optionnels)
            grid_metrics: métriques de la grille (difficulty, moyenne_cote_fav, etc.)

        Returns:
            dict {prob_1, prob_n, prob_2, prediction, confiance, probas}
        """
        w = self.weights.copy()

        # 1. Probabilités implicites des cotes (signal principal)
        cote_1 = match_data.get("cote_1", 0)
        cote_n = match_data.get("cote_n", 0)
        cote_2 = match_data.get("cote_2", 0)

        has_cotes = cote_1 > 0 and cote_n > 0 and cote_2 > 0
        if has_cotes:
            prob_cotes = np.array(self.odds_to_probabilities(cote_1, cote_n, cote_2))
        else:
            prob_cotes = np.array([1 / 3, 1 / 3, 1 / 3])

        # 2. Consensus joueurs (répartition)
        pct_1 = match_data.get("pct_1", 0)
        pct_n = match_data.get("pct_n", 0)
        pct_2 = match_data.get("pct_2", 0)

        total_pct = pct_1 + pct_n + pct_2
        has_consensus = total_pct > 0
        if has_consensus:
            prob_consensus = np.array([pct_1, pct_n, pct_2]) / total_pct
        else:
            prob_consensus = np.array([1 / 3, 1 / 3, 1 / 3])

        # 3. Signal Cyborg
        prono_cyborg = match_data.get("prono_cyborg", "")
        has_cyborg = bool(prono_cyborg and prono_cyborg.strip())
        prob_cyborg = self._parse_cyborg_signal(prono_cyborg)

        # 4. Ajustement difficulté
        profile = self.select_strategy_profile(grid_metrics)
        prob_difficulty = self.adjust_probabilities_for_difficulty(
            prob_cotes[0], prob_cotes[1], prob_cotes[2],
            grid_metrics, profile,
        )
        prob_difficulty = np.array(prob_difficulty)

        # 5. Biais historique
        prob_historique = HISTORICAL_1N2.copy()

        # --- Redistribution des poids si des signaux manquent ---
        # Si un signal n'est pas disponible, son poids est redistribué
        # proportionnellement aux signaux présents.
        if not has_cotes:
            w["cotes"] = 0.0
        if not has_consensus:
            w["consensus"] = 0.0
        if not has_cyborg:
            w["cyborg"] = 0.0

        total_w = sum(w.values())
        if total_w > 0 and total_w != 1.0:
            for k in w:
                w[k] /= total_w

        # --- Composition pondérée ---
        final = (
            w["cotes"] * prob_cotes
            + w["consensus"] * prob_consensus
            + w["cyborg"] * prob_cyborg
            + w["difficulte"] * prob_difficulty
            + w["historique"] * prob_historique
        )

        # Normaliser
        total = final.sum()
        if total > 0:
            final = final / total
        else:
            final = np.array([1 / 3, 1 / 3, 1 / 3])

        # Prediction et confiance
        pred_idx = int(np.argmax(final))
        prediction = ["1", "N", "2"][pred_idx]
        sorted_probs = np.sort(final)[::-1]
        confiance = float(sorted_probs[0] - sorted_probs[1])

        return {
            "prob_1": float(final[0]),
            "prob_n": float(final[1]),
            "prob_2": float(final[2]),
            "prediction": prediction,
            "confiance": confiance,
            "probas": {
                "1": float(final[0]),
                "N": float(final[1]),
                "2": float(final[2]),
            },
        }

    def predict_grid(self, grid_data: dict) -> list[dict]:
        """Prédit tous les matchs d'une grille.

        Args:
            grid_data: dict avec 'matches' et optionnellement des métriques
                       (difficulty, moyenne_cote_fav, etc.)

        Returns:
            liste de dicts de prédictions par match
        """
        matches = grid_data.get("matches", [])
        grid_metrics = {
            "difficulty": grid_data.get("difficulty"),
            "moyenne_cote_fav": grid_data.get("moyenne_cote_fav"),
            "somme_cotes_fav": grid_data.get("somme_cotes_fav"),
            "ecart_type_cotes": grid_data.get("ecart_type_cotes"),
            "inv_spread_sum": grid_data.get("inv_spread_sum"),
            "std_cote_fav": grid_data.get("std_cote_fav"),
            "accord_cotes_joueurs": grid_data.get("accord_cotes_joueurs"),
            "nb_matchs_serres": grid_data.get("nb_matchs_serres"),
        }

        # Compute metrics if not provided
        if grid_metrics["moyenne_cote_fav"] is None and matches:
            computed = self.compute_grid_metrics(matches)
            grid_metrics.update(computed)

        predictions = []
        for match in matches:
            pred = self.predict_match(match, grid_metrics)
            pred["home"] = match.get("home", "")
            pred["away"] = match.get("away", "")
            predictions.append(pred)

        return predictions

    @staticmethod
    def odds_to_probabilities(cote_1: float, cote_n: float,
                               cote_2: float) -> tuple[float, float, float]:
        """Convertit les cotes en probabilités implicites normalisées.

        Supprime la marge du bookmaker en normalisant à 1.

        Args:
            cote_1, cote_n, cote_2: cotes décimales

        Returns:
            tuple (prob_1, prob_n, prob_2) sommant à 1.0
        """
        if cote_1 <= 0 or cote_n <= 0 or cote_2 <= 0:
            return (1 / 3, 1 / 3, 1 / 3)

        raw_1 = 1.0 / cote_1
        raw_n = 1.0 / cote_n
        raw_2 = 1.0 / cote_2
        total = raw_1 + raw_n + raw_2

        if total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)

        return (raw_1 / total, raw_n / total, raw_2 / total)

    @staticmethod
    def compute_grid_metrics(matches: list) -> dict:
        """Calcule les métriques d'une grille à partir des cotes des matchs.

        Args:
            matches: liste de dicts avec cote_1, cote_n, cote_2

        Returns:
            dict avec moyenne_cote_fav, somme_cotes_fav, ecart_type_cotes
        """
        from statistics import mean, median, stdev

        cotes_fav = []
        for m in matches:
            cotes = [m.get("cote_1", 0), m.get("cote_n", 0), m.get("cote_2", 0)]
            valid = [c for c in cotes if c and c > 0]
            if valid:
                cotes_fav.append(min(valid))

        if not cotes_fav:
            return {
                "somme_cotes_fav": 0.0,
                "moyenne_cote_fav": 0.0,
                "mediane_cote_fav": 0.0,
                "ecart_type_cotes": 0.0,
            }

        return {
            "somme_cotes_fav": round(sum(cotes_fav), 2),
            "moyenne_cote_fav": round(mean(cotes_fav), 2),
            "mediane_cote_fav": round(median(cotes_fav), 2),
            "ecart_type_cotes": round(stdev(cotes_fav), 2) if len(cotes_fav) > 1 else 0.0,
        }

    def select_strategy_profile(self, grid_metrics: dict | None) -> str:
        """Selectionne le profil de strategie selon les metriques de la grille.

        Utilise en priorite inv_spread_sum (meilleur predicteur de surprises,
        effect size 1.24) et accord_cotes_joueurs, puis fallback sur la
        difficulte et la cote moyenne du favori.

        Args:
            grid_metrics: dict avec difficulty, moyenne_cote_fav,
                          inv_spread_sum, accord_cotes_joueurs, std_cote_fav

        Returns:
            "secure", "balanced" ou "risky"
        """
        if not grid_metrics:
            return "balanced"

        inv_spread = grid_metrics.get("inv_spread_sum")
        accord = grid_metrics.get("accord_cotes_joueurs")
        std_fav = grid_metrics.get("std_cote_fav")
        difficulty = grid_metrics.get("difficulty")
        moy_cote = grid_metrics.get("moyenne_cote_fav")
        t = self.difficulty_thresholds

        # inv_spread_sum < 2.0 => ~1 surprise => secure
        # inv_spread_sum > 3.0 => ~3 surprises => risky
        if inv_spread is not None:
            if inv_spread < 2.0:
                return "secure"
            if inv_spread > 3.5:
                return "risky"

        # Accord cotes/joueurs > 80% => les signaux convergent => secure
        if accord is not None and accord > 0.80:
            return "secure"

        # std_cote_fav < 0.26 (Q25%) => cotes homogenes => secure
        if std_fav is not None and std_fav < 0.26:
            return "secure"

        # Fallback sur difficulte + cote moyenne
        if (difficulty is not None and difficulty < t["secure_max_difficulty"]
                and moy_cote is not None and moy_cote < t["secure_max_cote_fav"]):
            return "secure"

        if difficulty is not None and difficulty > t["risky_min_difficulty"]:
            return "risky"

        if moy_cote is not None and moy_cote > t["risky_min_cote_fav"]:
            return "risky"

        return "balanced"

    def adjust_probabilities_for_difficulty(
        self, p1: float, pn: float, p2: float,
        grid_metrics: dict | None, profile: str,
    ) -> tuple[float, float, float]:
        """Ajuste les probabilités selon le profil de difficulté.

        Args:
            p1, pn, p2: probabilités initiales
            grid_metrics: métriques de la grille
            profile: "secure", "balanced" ou "risky"

        Returns:
            tuple (p1_adj, pn_adj, p2_adj) normalisées
        """
        probs = np.array([p1, pn, p2])

        if profile == "secure":
            # Renforcer le favori
            max_idx = np.argmax(probs)
            boost = np.zeros(3)
            boost[max_idx] = 0.05
            probs = probs + boost
        elif profile == "risky":
            # Aplatir les probabilités vers l'uniforme
            uniform = np.array([1 / 3, 1 / 3, 1 / 3])
            probs = 0.8 * probs + 0.2 * uniform
        # "balanced" : pas d'ajustement

        total = probs.sum()
        if total > 0:
            probs = probs / total

        return (float(probs[0]), float(probs[1]), float(probs[2]))

    def _parse_cyborg_signal(self, prono: str) -> np.ndarray:
        """Convertit le prono Cyborg en vecteur de probabilité.

        Args:
            prono: texte du prono ("1", "N", "2", "1N", "N2", "12", "1N2")

        Returns:
            array de probabilités [p1, pn, p2]
        """
        if not prono:
            return np.array([1 / 3, 1 / 3, 1 / 3])

        prono = prono.strip().upper()
        signal = np.array([0.0, 0.0, 0.0])
        mapping = {"1": 0, "N": 1, "2": 2}

        for char in prono:
            if char in mapping:
                signal[mapping[char]] = 1.0

        total = signal.sum()
        if total == 0:
            return np.array([1 / 3, 1 / 3, 1 / 3])

        # Single outcome: high confidence
        if total == 1:
            idx = np.argmax(signal)
            result = np.array([0.10, 0.10, 0.10])
            result[idx] = 0.80
            return result

        # Two outcomes: moderate confidence
        if total == 2:
            result = np.array([0.10, 0.10, 0.10])
            for i in range(3):
                if signal[i] > 0:
                    result[i] = 0.45
            return result

        # All three: uniform
        return np.array([1 / 3, 1 / 3, 1 / 3])
