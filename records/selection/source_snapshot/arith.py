"""Lightweight arithmetic verifier over candidate reasoning text.

The official scorer only grades the final number, but a candidate whose intermediate
equations do not add up is much less likely to have reached the right number by a
valid route. This module re-evaluates the arithmetic statements that appear in a
solution and reports how many of them actually hold. It is deliberately symbolic and
safe: only digits, operators and parentheses are ever evaluated.
"""
import re
from fractions import Fraction

_FRAC_RE = re.compile(r'\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
_TEXT_RE = re.compile(r'\\(?:text|mbox|mathrm|operatorname)\s*\{[^{}]*\}')
_BOXED_RE = re.compile(r'\\boxed')
_NUM_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')
_EXPR_CHARS = re.compile(r'[^0-9\.\+\-\*/\(\)\s]')
_EQ_SPLIT = re.compile(r'(?<![=<>])=(?![=>])')
_WORD_OPS = [('×', '*'), ('÷', '/'), ('·', '*'), ('\\times', '*'), ('\\cdot', '*'),
             ('\\div', '/'), ('−', '-'), ('–', '-'), ('—', '-')]
_LINE_SPLIT = re.compile(r'[\n;]|\|{|\\\\')
_WORD_RE = re.compile(r'[A-Za-z]{2,}')
_VAR_RE = re.compile(r'(?:\d\s*[A-Za-z])|(?:[A-Za-z]\s*\d)')


def _clean(text):
    t = text
    t = _TEXT_RE.sub(' ', t)
    t = _FRAC_RE.sub(lambda m: '((%s)/(%s))' % (m.group(1), m.group(2)), t)
    for a, b in _WORD_OPS:
        t = t.replace(a, b)
    t = _BOXED_RE.sub(' ', t)
    t = t.replace('$', ' ').replace('%', ' /100 ')
    t = re.sub(r'\\left|\\right|\\!|\\,|\\;|\\quad|\\%', ' ', t)
    t = t.replace('{', '(').replace('}', ')')
    return t


def _side_ok(raw):
    """Return cleaned arithmetic string, or None if the side is not pure arithmetic."""
    s = _WORD_RE.sub(' ', raw)          # drop multi-letter words (units, labels)
    if _VAR_RE.search(s):               # variable next to a digit, e.g. '4x'
        return None
    s = _EXPR_CHARS.sub(' ', s).strip()
    s = re.sub(r'\s+', ' ', s)
    if not re.search(r'\d', s):
        return None
    if len(s) > 120:
        return None
    return s


def _eval(expr):
    expr = expr.strip()
    if not expr:
        return None
    tokens = re.findall(r'\d[\d,]*(?:\.\d+)?|[-+*/()]', expr)
    if not tokens:
        return None
    depth, out = 0, []
    for tk in tokens:
        if tk == '(':
            depth += 1
        elif tk == ')':
            if depth == 0:
                continue
            depth -= 1
        out.append(tk)
    out.append(')'*depth)
    py = re.sub(r'\d[\d,]*(?:\.\d+)?',
                lambda m: "Fraction('%s')" % m.group().replace(',', ''), ' '.join(out))
    py = py.replace('^', '**')
    try:
        value = eval(py, {'Fraction': Fraction, '__builtins__': {}})  # noqa: S307
    except Exception:
        return None
    return value if isinstance(value, Fraction) else None


def _is_rounded(expr):
    """True if any literal has 2-4 decimals, i.e. a rounded value."""
    return bool(re.search(r'\d\.\d{2,4}(?!\d)', expr))


def _equal(a, b, a_expr, b_expr):
    if a == b:
        return True
    if _is_rounded(a_expr) or _is_rounded(b_expr):
        fa, fb = float(a), float(b)
        if fb == 0 or fa == 0:
            return False
        return abs(fa-fb)/max(abs(fb), 1e-9) < 2e-3
    return False


def check_equations(text):
    """Return (n_checked, n_ok, last_ok, last_rhs)."""
    t = _clean(text)
    n_checked = n_ok = 0
    last_ok = None
    last_rhs = None
    for line in _LINE_SPLIT.split(t):
        if '=' not in line:
            continue
        parts = _EQ_SPLIT.split(line)
        if len(parts) < 2:
            continue
        for k in range(len(parts)-1):
            rhs_raw = parts[k+1]
            # skip mid-chain comparisons where the right side is not yet evaluated
            if k+1 < len(parts)-1 and re.search(r'[-+*/]', _clean_side_chars(rhs_raw)):
                continue
            lhs = _side_ok(parts[k])
            rhs = _side_ok(rhs_raw)
            if lhs is None or rhs is None:
                continue
            lv, rv = _eval(lhs), _eval(rhs)
            if lv is None or rv is None:
                continue
            n_checked += 1
            ok = _equal(lv, rv, lhs, rhs)
            n_ok += int(ok)
            last_ok = int(ok)
            last_rhs = rv
    return n_checked, n_ok, last_ok, last_rhs


def _clean_side_chars(raw):
    return _EXPR_CHARS.sub(' ', _WORD_RE.sub(' ', raw))


def verify(text, final_value=None):
    n_checked, n_ok, last_ok, last_rhs = check_equations(text)
    f = {
        'ar_n_checked': float(n_checked),
        'ar_n_ok': float(n_ok),
        'ar_n_bad': float(n_checked-n_ok),
        'ar_frac_ok': (n_ok/n_checked) if n_checked else 0.5,
        'ar_has_check': float(n_checked > 0),
        'ar_last_ok': float(last_ok) if last_ok is not None else 0.0,
        'ar_all_ok': float(bool(n_checked) and n_ok == n_checked),
        'ar_any_bad': float(n_checked > 0 and n_ok < n_checked),
    }
    if final_value is not None and last_rhs is not None:
        f['ar_ans_last_rhs'] = float(abs(float(last_rhs)-float(final_value)) < 1e-6)
        f['ar_gap_last'] = float(abs(float(last_rhs)-float(final_value)))
    else:
        f['ar_ans_last_rhs'] = 0.0
        f['ar_gap_last'] = 0.0
    return f


def _to_frac(tok):
    tok = tok.replace(',', '').strip()
    try:
        return Fraction(tok)
    except (ValueError, ZeroDivisionError):
        return None


def question_number_coverage(question, text):
    qnums = [v for v in (_to_frac(x) for x in _NUM_RE.findall(question)) if v is not None]
    if not qnums:
        return {'qn_n': 0.0, 'qn_cov': 0.0, 'qn_missing': 0.0, 'qn_extra': 0.0}
    tnums = {v for v in (_to_frac(x) for x in _NUM_RE.findall(text)) if v is not None}
    used = sum(int(q in tnums) for q in qnums)
    return {'qn_n': float(len(qnums)), 'qn_cov': used/len(qnums),
            'qn_missing': float(len(qnums)-used),
            'qn_extra': float(sum(int(t not in qnums) for t in tnums))}
