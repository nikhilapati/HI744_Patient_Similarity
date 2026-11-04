"""HI 744 retrieval component."""
import json
from pathlib import Path
from scipy import sparse
from .data_loader import load_patients
from .tfidf_model import build_representations
from .similarity import retrieve_tfidf
from .bm25_model import retrieve_bm25
from .evaluation import evaluate


def run(input_dir, output_dir, max_features=30000):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    patients = load_patients(input_dir)
    texts, vectorizer, counts, tfidf = build_representations(patients, max_features)
    ids = [str(p['patient_uid']) for p in patients]
    # Load with scipy.sparse.load_npz; row index and term index maps in metadata JSON.
    sparse.save_npz(out / 'tfidf_matrix.npz', tfidf)
    (out / 'tfidf_metadata.json').write_text(json.dumps({
        'patient_uids': ids, 'vocabulary': {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        'settings': {'smooth_idf': True, 'sublinear_tf': True, 'norm': 'l2',
                     'max_features': max_features, 'text_field': 'patient'}
    }))
    t = retrieve_tfidf(tfidf)
    b = retrieve_bm25(counts)
    rankings = {uid: {'patient_id': str(p['patient_id']),
                      'tfidf': [ids[j] for j in t[i]],
                      'bm25': [ids[j] for j in b[i]]}
                for i, (uid, p) in enumerate(zip(ids, patients))}
    (out / 'patient_similarity.json').write_text(json.dumps(rankings, indent=2))
    metrics, summary = evaluate(patients, rankings)
    (out / 'performance.json').write_text(json.dumps(metrics, indent=2))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
