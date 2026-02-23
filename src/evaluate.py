from typing import List, Dict


def retrieval_stats(retrieved: List[Dict]) -> Dict:
    if not retrieved:
        return {"top_score": 0.0, "avg_score": 0.0}

    scores = [r["score"] for r in retrieved]
    return {
        "top_score": float(max(scores)),
        "avg_score": float(sum(scores) / len(scores)),
        "doc_ids": list({r["doc_id"] for r in retrieved}),
    }


def simple_grounding_check(answer: str, retrieved: List[Dict]) -> Dict:
    """
    Very simple heuristic:
    - checks if any doc_id is mentioned in answer (encouraged by prompt)
    """
    doc_ids = {r["doc_id"] for r in retrieved}
    mentioned = [d for d in doc_ids if d in answer]
    return {
        "docs_mentioned": mentioned,
        "mentioned_count": len(mentioned),
        "retrieved_doc_count": len(doc_ids),
    }