"""HI 744 retrieval component."""
import numpy as np


def retrieve_bm25(counts, k=5, k1=1.5, b=0.75):
    """Sparse Okapi BM25: query is the patient's distinct terms."""
    n = counts.shape[0]
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    avgdl = max(float(lengths.mean()), 1e-9)
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
    weighted = counts.copy().astype(np.float32)
    weighted.data = weighted.data * (k1 + 1) / (
        weighted.data + k1 * (1 - b + b * np.repeat(lengths, np.diff(weighted.indptr)) / avgdl)
    )
    weighted = weighted.multiply(idf.astype(np.float32)).tocsr()
    binary_queries = counts.copy()
    binary_queries.data[:] = 1
    results = []
    # Sparse matrix multiplication in blocks avoids dense n-by-n allocation.
    for start in range(0, n, 128):
        block = (binary_queries[start:start+128] @ weighted.T).tocsr()
        for local in range(block.shape[0]):
            i = start + local
            row = block.getrow(local)
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
