"""HI 744 retrieval component."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def preprocess(text):
    """Lowercase, strip punctuation, remove English stopwords, lightly lemmatize.

    Rule-based lemmatization is intentionally conservative for clinical vocabulary;
    a full biomedical lemmatizer is an optional enhancement.
    """
    tokens = re.findall(r'[a-z]+(?:\d+)?|\d+(?:\.\d+)?', str(text).lower())
    out = []
    for t in tokens:
        if t in ENGLISH_STOP_WORDS or len(t) == 1:
            continue
        if len(t) > 5 and t.endswith('ies'):
            t = t[:-3] + 'y'
        elif len(t) > 5 and t.endswith('sses'):
            t = t[:-2]
        elif len(t) > 5 and t.endswith('s') and not t.endswith(('ss', 'us', 'is')):
            t = t[:-1]
        out.append(t)
    return ' '.join(out)
