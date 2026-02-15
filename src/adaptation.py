from .case_library import CaseFeatures, CaseSolution, CLASSIC_SYMPTOMS
from .retrieval import retrieve_top_k


_SIMILARITY_THRESHOLD_DIRECT = 0.88


def adapt_solution(
    query: CaseFeatures,
    retrieved: list[dict],
    rule_wm: dict,
) -> dict:
    """
    Reuse step of the CBR cycle.

    Strategy:
    - If the best match exceeds SIMILARITY_THRESHOLD_DIRECT, adopt its solution
      with minor adjustments.
    - Otherwise, build a solution from the rule engine's working memory,
      using the top retrieved case only for confidence estimation.

    The rule_wm (working memory from rule_engine.run_rules) always acts as a
    safety layer: any emergency or discordance flags it raises override the
    retrieved solution.

    Returns a plain dict representing the adapted solution.
    """
    if not retrieved:
        return _solution_from_rules(rule_wm, base_confidence=0.50)

    best = retrieved[0]
    sim = best["similarity"]
    base_sol: CaseSolution = best["case"].solution

    if sim >= _SIMILARITY_THRESHOLD_DIRECT:
        adapted = {
            "diagnosis": base_sol.diagnosis,
            "status": base_sol.status,
            "classification": base_sol.classification,
            "recommendations": list(base_sol.recommendations),
            "confidence": sim,
            "source": "retrieved",
            "source_case_id": best["case"].id,
        }
    else:
        adapted = _solution_from_rules(rule_wm, base_confidence=sim)
        adapted["source"] = "rule-derived"
        adapted["source_case_id"] = best["case"].id

    adapted = _apply_emergency_overrides(adapted, rule_wm, query)
    adapted = _apply_discordance_overrides(adapted, rule_wm)

    return adapted


def _solution_from_rules(wm: dict, base_confidence: float) -> dict:
    """Constructs a solution dict directly from working memory assertions."""
    diagnosis = "Inconclusive"
    classification = wm.get("classification") or "unclassified"

    if wm.get("emergency_dka"):
        diagnosis = "Diabetes Mellitus with DKA"
    elif wm.get("confirmed_diabetes"):
        if classification == "suspected-type-1":
            diagnosis = "Type 1 Diabetes Mellitus"
        elif classification == "suspected-type-2":
            diagnosis = "Type 2 Diabetes Mellitus"
        else:
            diagnosis = "Diabetes Mellitus (Unclassified)"
    elif wm.get("discordant_results"):
        diagnosis = "Inconclusive - Discordant Results"
    elif wm.get("provisional_hyperglycemia"):
        diagnosis = "Provisional Hyperglycemia"

    recommendations = list(wm.get("recommendations", []))
    if wm.get("confirmed_diabetes") and "suspected-type-2" in classification:
        recommendations += ["lifestyle-modifications", "metformin-initiation"]

    return {
        "diagnosis": diagnosis,
        "status": wm.get("status", "Unknown"),
        "classification": classification,
        "recommendations": list(dict.fromkeys(recommendations)),
        "confidence": base_confidence,
        "source": "rule-derived",
        "source_case_id": None,
    }


def _apply_emergency_overrides(adapted: dict, wm: dict, query: CaseFeatures) -> dict:
    """
    DKA is a safety-critical rule. If the rule engine fires the emergency rule,
    it overrides any retrieved solution that didn't flag it.
    """
    if wm.get("emergency_dka") and "Emergency" not in adapted.get("status", ""):
        adapted["status"] = "Medical Emergency - Potential DKA"
        adapted["diagnosis"] = "Diabetes Mellitus with DKA"
        recs = set(adapted.get("recommendations", []))
        recs.update(["refer-to-er", "immediate-insulin-therapy"])
        adapted["recommendations"] = list(recs)
    return adapted


def _apply_discordance_overrides(adapted: dict, wm: dict) -> dict:
    """
    Discordant results (Rule 6) block confirmation and mandate repeat testing
    regardless of what the retrieved case suggested.
    """
    if wm.get("discordant_results") and not wm.get("confirmed_diabetes"):
        adapted["diagnosis"] = "Inconclusive - Discordant Results"
        adapted["status"] = "Discordant Results - Repeat Required"
        recs = set(adapted.get("recommendations", []))
        recs.add("repeat-fpg")
        adapted["recommendations"] = list(recs)
        adapted["confidence"] = min(adapted.get("confidence", 0.5), 0.65)
    return adapted
