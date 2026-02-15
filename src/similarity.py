from case_library import CaseFeatures, FEATURE_WEIGHTS, NUMERIC_RANGES


def _normalize(value: float, feature: str) -> float:
    """
    Maps a numeric feature value to [0, 1] using clinical range boundaries
    defined in NUMERIC_RANGES.
    """
    low, high = NUMERIC_RANGES[feature]
    return max(0.0, min(1.0, (value - low) / (high - low)))


def local_similarity_numeric(feature: str, val_q: float, val_c: float) -> float:
    """
    Computes similarity between two numeric feature values as 1 - |diff|
    in normalized space. Returns 0.5 (neutral) when either value is absent.
    """
    if val_q is None or val_c is None:
        return 0.5
    norm_q = _normalize(val_q, feature)
    norm_c = _normalize(val_c, feature)
    return 1.0 - abs(norm_q - norm_c)


def local_similarity_ketones(val_q: str, val_c: str) -> float:
    """
    Ketones similarity: exact match = 1.0, positive vs negative = 0.0,
    trace counts as partial (0.5) against both extremes.
    """
    if val_q is None or val_c is None:
        return 0.5
    if val_q == val_c:
        return 1.0
    trace_set = {"trace", "positive"}
    if val_q in trace_set and val_c in trace_set:
        return 0.6
    return 0.0


def local_similarity_symptoms(has_q: bool, has_c: bool) -> float:
    """Boolean exact match for presence of classic symptoms."""
    return 1.0 if has_q == has_c else 0.0


def global_similarity(query: CaseFeatures, case: CaseFeatures) -> float:
    """
    Computes weighted global similarity between a query and a stored case.
    Uses the clinical feature weights from FEATURE_WEIGHTS.

    Returns a float in [0, 1] where 1.0 = perfect match.
    """
    numeric_features = ["age", "bmi", "fpg", "a1c", "random_glucose"]

    scores = {}

    for feat in numeric_features:
        q_val = getattr(query, feat)
        c_val = getattr(case, feat)
        scores[feat] = local_similarity_numeric(feat, q_val, c_val)

    scores["ketones"] = local_similarity_ketones(query.ketones, case.ketones)
    scores["has_classic_symptoms"] = local_similarity_symptoms(
        query.has_classic_symptoms, case.has_classic_symptoms
    )

    total_weight = sum(FEATURE_WEIGHTS.values())
    weighted_sum = sum(
        FEATURE_WEIGHTS[feat] * scores[feat] for feat in FEATURE_WEIGHTS
    )

    return weighted_sum / total_weight
