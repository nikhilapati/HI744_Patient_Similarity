# HI 744 — Patient Similarity Retrieval

A reproducible implementation of the Fall 2026 programming assignment using the supplied PMC-Patients subset (8,579 patient JSON records).

## Quick start

```bash
python -m pip install -r requirements.txt
python patient_similarity_retrieval.py data/PMC_Patients
```

This creates `outputs/tfidf_matrix.npz`, `outputs/tfidf_metadata.json`, `outputs/patient_similarity.json`, `outputs/performance.json`, and `outputs/summary.json`.

## Inspect a TF-IDF weight

```python
import json
from scipy.sparse import load_npz
m = json.load(open('outputs/tfidf_metadata.json'))
X = load_npz('outputs/tfidf_matrix.npz')
uid, term = '6822881-4', 'pulmonary'
print(float(X[m['patient_uids'].index(uid), m['vocabulary'][term]]))
```

## Methods

- Clinical note text: lowercase, punctuation removal, English stopword removal, conservative rule-based word normalization/lemmatization.
- TF-IDF: scikit-learn, smoothed IDF, sublinear TF, L2 normalization, max 30,000 terms, cosine similarity.
- BM25: Okapi BM25, k1=1.5, b=0.75; each document's unique tokens serve as query terms.
- Excludes the query patient from its own top 5. Deterministic tie-breaking by corpus index.
- Relevance 1 and 2 both count as relevant. Recall is null for queries with no relevant labels; these are excluded from mean recall.

## Limitations

The supplied ZIP is a subset of the full PMC-Patients dataset. Many annotated similar patients are absent from this candidate pool, which limits achievable recall. The assignment-defined recall denominator includes all annotated relevant patients, even if absent from the corpus. The lemmatizer is intentionally conservative and not a full linguistic/clinical lemmatizer. The system is an academic text-retrieval exercise, not a validated clinical decision support tool.

## Repo layout

- `src/`: separate data loading, preprocessing, TF-IDF, BM25, ranking, evaluation, and pipeline modules
- `patient_similarity_retrieval.py`: assignment-compatible command
- `outputs/`: machine-readable outputs and metrics
- `report.pdf`: short methods/results report
- `data/PMC_Patients/`: supplied input data
