"""HI 744 retrieval component."""
import numpy as np
from .utils import parse_relevance


def evaluate(patients, rankings):
    known = {str(p['patient_uid']) for p in patients}
    metrics = {}
    summary = {}
    for method in ('tfidf', 'bm25'):
        precision, recall, overlaps = [], [], []
        excluded = 0
        for i, p in enumerate(patients):
            uid = str(p['patient_uid'])
            relevant = parse_relevance(p.get('similar_patients')) - {uid}
            found = set(rankings[uid][method]) & relevant
            pr = len(found) / 5
            rc = len(found) / len(relevant) if relevant else None
            metrics.setdefault(uid, {'patient_id': str(p['patient_id'])})[method] = {
                'precision_5': pr, 'recall_5': rc}
            precision.append(pr)
            if rc is None:
                excluded += 1
            else:
                recall.append(rc)
        summary[method] = {'mean_precision_5': float(np.mean(precision)),
                           'mean_recall_5': float(np.mean(recall)) if recall else None,
                           'recall_excluded': excluded}
    summary['mean_top5_overlap'] = float(np.mean([
        len(set(r['tfidf']) & set(r['bm25'])) / 5 for r in rankings.values()]))
    summary['dataset_size'] = len(patients)
    summary['mean_words_raw'] = float(np.mean([len(str(p['patient']).split()) for p in patients]))
    summary['relevance_links_outside_corpus'] = sum(len(parse_relevance(p.get('similar_patients')) - known)
                                                   for p in patients)
    return metrics, summary
