from case_library import CaseFeatures, CaseLibrary, Case
from similarity import global_similarity


def retrieve_top_k(
    query_features: CaseFeatures,
    library: CaseLibrary,
    k: int = 3,
) -> list[dict]:
    """
    Computes global similarity between the query and every case in the library,
    then returns the top-k most similar cases sorted descending by similarity.

    Each element in the returned list is:
        {"case": Case, "similarity": float}
    """
    scored = [
        {"case": case, "similarity": global_similarity(query_features, case.features)}
        for case in library.all_cases()
    ]
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:k]


def best_match(retrieved: list[dict]) -> dict | None:
    """Returns the single highest-similarity result, or None if list is empty."""
    return retrieved[0] if retrieved else None
