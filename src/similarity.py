"""HI 744 retrieval component."""
import numpy as np


def retrieve_tfidf(tfidf, k=5):
    scores = (tfidf @ tfidf.T).tocsr()
    n = tfidf.shape[0]
    results = []
    for i in range(n):
        row = scores.getrow(i)
        idx, vals = row.indices, row.data
        mask = idx != i
        idx, vals = idx[mask], vals[mask]
        order = np.lexsort((idx, -vals))[:k]
        selected = idx[order].tolist()
        if len(selected) < k:
            present = set(selected) | {i}
            selected.extend(j for j in range(n) if j not in present and len(selected) < k)
        results.append(selected)
    return results
