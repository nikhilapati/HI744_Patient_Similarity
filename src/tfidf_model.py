"""HI 744 retrieval component."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from scipy import sparse
from .preprocessing import preprocess


def build_representations(patients, max_features=30000):
    texts = [preprocess(p['patient']) for p in patients]
    cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=max_features, dtype=np.float32)
    counts = cv.fit_transform(texts).tocsr()
    transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf = transformer.fit_transform(counts).tocsr()
    return texts, cv, counts, tfidf
