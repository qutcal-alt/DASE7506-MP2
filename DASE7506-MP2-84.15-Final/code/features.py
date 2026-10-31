"""Candidate feature extraction for MP2 select-the-best-solution.

Every feature is computed from the question text, the four candidate texts and the
supplied 1024-d whole-trace representations only. No reference answers are touched.
"""
import re
from fractions import Fraction

import numpy as np

from arith import verify

from common import parse_answer


def question_number_coverage_frac(question, text):
    from arith import question_number_coverage
    return question_number_coverage(question, text)

NUM_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')
OP_RE = re.compile(r'[-+*/÷×=]')
LINE_RE = re.compile(r'\n')

# Discourse / metacognitive markers that often accompany self-correction or padding.
MARKERS = {
    'wait': r'\bwait\b',
    'hmm': r'\bhmm\b',
    'actually': r'\bactually\b',
    'instead': r'\binstead\b',
    'mistake': r'\bmistake\b',
    'recheck': r'\bre-?check\b',
    'verify': r'\bverif',
    'however': r'\bhowever\b',
    'therefore': r'\btherefore\b',
    'thus': r'\bthus\b',
    'so_comma': r'\bso\b',
    'lets': r"\blet'?s\b",
    'step': r'\bstep\b',
    'answer_kw': r'\banswer\b',
    'final': r'\bfinal\b',
    'total': r'\btotal\b',
    'approx': r'\bapproxim|\babout\b|\baround\b|\broughly\b',
    'bold': r'\*\*',
    'latex': r'\$',
    'frac': r'\\frac|\\d?frac',
    'boxed': r'\\boxed',
    'hash4': r'####',
    'bullet': r'^\s*[-*·]',
    'question_mark': r'\?',
    'unit_word': r'\b(miles?|dollars?|cents?|apples?|hours?|minutes?|days?|weeks?|months?|years?|percent|%)\b',
    'paren_math': r'\([^)]*\d[^)]*\)',
}
MARKER_KEYS = sorted(MARKERS)


def _safe_frac(text):
    return parse_answer(text)


def text_features(question, text):
    n_char = len(text)
    words = text.split()
    n_word = len(words)
    nums = NUM_RE.findall(text)
    n_num = len(nums)
    lines = [l for l in LINE_RE.split(text) if l.strip()]
    n_line = len(lines)
    n_eq = text.count('=')
    n_op = len(OP_RE.findall(text))
    f = {
        'len_char': n_char,
        'len_word': n_word,
        'log_len': np.log1p(n_char),
        'n_num': n_num,
        'n_line': n_line,
        'n_eq': n_eq,
        'n_op': n_op,
        'num_per_line': n_num/max(n_line, 1),
        'num_per_word': n_num/max(n_word, 1),
        'op_per_num': n_op/max(n_num, 1),
        'avg_word_len': n_char/max(n_word, 1),
        'digit_ratio': sum(len(x) for x in nums)/max(n_char, 1),
        'eq_per_line': n_eq/max(n_line, 1),
        'tail_digits': len(NUM_RE.findall(text[-120:])),
    }
    low = text.lower()
    for k in MARKER_KEYS:
        f['m_'+k] = len(re.findall(MARKERS[k], low, flags=re.M))
    # normalised marker rates
    for k in MARKER_KEYS:
        f['r_'+k] = f['m_'+k]/max(n_word, 1)
    return f


def answer_value(text):
    a = parse_answer(text)
    if a is None:
        return None
    try:
        return float(Fraction(a))
    except Exception:
        return None


def build_features(bank, arrays):
    """Return (X, names, meta). X has shape [N, 4, F]; meta holds answer info."""
    E = np.asarray(arrays['embeddings'], dtype=np.float32)
    N = len(bank)
    En = E/np.linalg.norm(E, axis=-1, keepdims=True).astype(np.float32)
    centroid = En.mean(axis=1, keepdims=True)
    cen = centroid/np.linalg.norm(centroid, axis=-1, keepdims=True)
    cos_cent = np.einsum('nkd,nkd->nk', En, cen.astype(np.float32))
    # mean cosine to the other three candidates
    sim_all = np.einsum('nkd,nmd->nkm', En, En)
    off = sim_all - np.eye(4)[None]*np.diagonal(sim_all, axis1=1, axis2=2)[:, :, None]
    mean_other = off.sum(-1)/3.0

    raw_texts = [[c['text'] for c in r['candidates']] for r in bank]
    q_lens = np.array([[len(question_numbers(r['question']))] for r in bank], dtype=np.float32)

    rows = []
    for i, r in enumerate(bank):
        q = r['question']
        qnums = question_numbers(q)
        feats = [text_features(q, t) for t in raw_texts[i]]
        answers = [_safe_frac(t) for t in raw_texts[i]]
        vals = [answer_value(t) for t in raw_texts[i]]
        arith = [verify(t, vals[j]) for j, t in enumerate(raw_texts[i])]
        qcov = [question_number_coverage_frac(q, t) for t in raw_texts[i]]
        # self-consistency structure
        counts = {}
        for a in answers:
            if a is not None:
                counts[a] = counts.get(a, 0) + 1
        n_distinct = len(counts)
        modal_n = max(counts.values()) if counts else 0
        for j, f in enumerate(feats):
            a = answers[j]
            v = vals[j]
            g = {
                'agree_count': counts.get(a, 0) if a is not None else 0,
                'agree_frac': (counts.get(a, 0)/4.0) if a is not None else 0.0,
                'is_modal': int(bool(a is not None and counts.get(a, 0) == modal_n)),
                'modal_n': modal_n,
                'n_distinct': n_distinct,
                'has_answer': int(a is not None),
                'ans_mag': np.log1p(abs(v)) if v is not None else 0.0,
                'ans_neg': float(v < 0) if v is not None else 0.0,
                'ans_int': float(v is not None and abs(v - round(v)) < 1e-9),
                'ans_in_text': int(bool(v is not None and any(
                    abs(float(Fraction(x.replace(',', '')))) == abs(v)
                    for x in NUM_RE.findall(raw_texts[i][j][:-60]) ))) if False else 0,
                'ans_gt_maxq': float(bool(v is not None and qnums and v > max(qnums))),
                'ans_lt_minq': float(bool(v is not None and qnums and v < min(qnums))),
                'ans_eq_anyq': float(bool(v is not None and any(abs(v - x) < 1e-9 for x in qnums))),
                'ans_ratio_maxq': (v/max(qnums) if (v is not None and qnums and max(qnums) > 0) else 0.0),
                'n_qnums': len(qnums),
                'cos_cent': float(cos_cent[i, j]),
                'mean_other_cos': float(mean_other[i, j]),
            }
            # group-level arithmetic-consistency context
            g.update({
                'ar_all_ok_n': sum(int(a['ar_all_ok']) for a in arith),
                'ar_all_ok_minus': sum(int(a['ar_all_ok']) for a in arith) - arith[j]['ar_all_ok'],
                'ar_ans_rhs_n': sum(int(a['ar_ans_last_rhs']) for a in arith),
            })
            rows.append({**f, **arith[j], **qcov[j], **g})

    names = sorted(rows[0].keys())
    X = np.array([[row[k] for k in names] for row in rows], dtype=np.float64).reshape(N, 4, -1)
    # group-relative versions of a few features
    idxs = {n: k for k, n in enumerate(names)}
    add = []
    addnames = []
    for key in ('len_char', 'len_word', 'n_num', 'n_line', 'n_eq', 'n_op', 'cos_cent',
                'mean_other_cos', 'agree_count', 'ans_mag', 'ar_frac_ok', 'ar_all_ok',
                'ar_n_checked', 'ar_ans_last_rhs', 'qn_missing', 'qn_cov'):
        col = X[:, :, idxs[key]]
        mu = col.mean(1, keepdims=True)
        sd = col.std(1, keepdims=True) + 1e-6
        add.append(((col-mu)/sd)[:, :, None])
        addnames.append('z_'+key)
        add.append((col-mu)[:, :, None])
        addnames.append('d_'+key)
        rank = np.argsort(np.argsort(col, axis=1), axis=1).astype(np.float64)
        add.append(rank[:, :, None])
        addnames.append('rk_'+key)
    X = np.concatenate([X]+add, axis=-1)
    names = names + addnames
    meta = {'answers': [[_safe_frac(t) for t in row] for row in raw_texts],
            'values': [[answer_value(t) for t in row] for row in raw_texts]}
    return X, names, meta


def question_numbers(q):
    out = []
    for x in NUM_RE.findall(q):
        try:
            out.append(float(x.replace(',', '')))
        except ValueError:
            pass
    return out
