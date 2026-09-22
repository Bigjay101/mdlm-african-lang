import re
def artifact_rate(text):
    """Single letters stranded between spaces -- the 'b a r a b a' pattern."""
    tokens = text.split()
    singles = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    return singles, len(tokens), 1000 * singles / max(len(tokens), 1)