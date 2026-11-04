"""HI 744 retrieval component."""
import ast


def parse_relevance(raw):
    if isinstance(raw, dict):
        value = raw
    elif isinstance(raw, str):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return set()
    else:
        return set()
    return {str(k) for k, v in value.items() if str(v) in ('1', '2')}
