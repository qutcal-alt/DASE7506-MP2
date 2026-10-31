"""Soft arithmetic provenance features using candidate inputs only.

Unlike whole-text number coverage, these distinguish a number repeated in prose
from a number actually used on a verified path to the parsed final answer.
This is deliberately an approximate feature extractor, not a semantic verifier:
equal-valued quantities are conflated, implicit quantities may be missing, and
unnecessary question numbers can be legitimate. No feature vetoes an answer.
"""
import ast
import math
import re
from fractions import Fraction

import numpy as np

from common import parse_answer


FEATURE_NAMES = (
    'q_value_count_log',
    'verified_equation_count_log',
    'verified_equation_fraction',
    'q_values_used_in_verified_arithmetic_fraction',
    'q_values_in_final_ancestry_fraction',
    'q_values_used_outside_final_ancestry_fraction',
    'q_values_mentioned_outside_final_ancestry_fraction',
    'has_verified_final_trace',
    'final_trace_depth_log',
    'maximum_trace_depth_log',
    'final_unknown_ancestor_fraction',
    'all_unknown_operand_fraction',
    'final_known_ancestor_count_log',
    'unused_computed_value_fraction',
    'final_ancestry_fraction_of_arithmetic_q_values',
    'q_values_used_more_than_once_fraction',
    'unverified_equation_count_log',
    'has_parsed_official_answer',
)
DIM = len(FEATURE_NAMES)
_NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?')
_FRAC = re.compile(r'\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
_RUN_LEFT = re.compile(r'[\d.()+*/\-\s]+$')
_RUN_RIGHT = re.compile(r'^[\d.()+*/\-\s]+')
_WORDS = dict(zip(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),
    range(20)))
_WORDS.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10))))
_WORD_PATTERN = re.compile(r'\b(' + '|'.join(_WORDS) + r')(?:[ -]+(' + '|'.join(list(_WORDS)[1:10]) + r'))?\b', re.I)


def _key(value):
    return round(float(value), 6)


def _normalize(text):
    text = str(text).replace(',', '').replace('**', '')
    text = text.replace('\u2212', '-').replace('\u00d7', '*').replace('\u00f7', '/')
    text = re.sub(r'\\(?:times|cdot)', '*', text)
    text = re.sub(r'\\div', '/', text)
    text = re.sub(r'\\(?:left|right)', '', text)
    text = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', ' ', text)
    for _ in range(4):
        text, count = _FRAC.subn(lambda m: '((' + m[1] + ')/(' + m[2] + '))', text)
        if not count:
            break
    text = re.sub(r'\\boxed\{([^{}]*)\}', r'(\1)', text)
    text = re.sub(r'\\[!;,\[\]()]', ' ', text)
    return text.replace('$', '').replace('\\%', '%')


def _numbers(text, words=False):
    values = set()
    for match in _NUMBER.finditer(str(text).replace(',', '')):
        try:
            parts = match[0].replace(' ', '').split('/')
            value = Fraction(parts[0])
            if len(parts) == 2:
                value /= Fraction(parts[1])
            if abs(value) <= 1e12:
                values.add(_key(value))
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    if words:
        for match in _WORD_PATTERN.finditer(str(text)):
            first = _WORDS[match[1].lower()]
            second = _WORDS[match[2].lower()] if match[2] else 0
            # Combine twenty-one; keep "one two" as two separate values.
            if first >= 20:
                values.add(float(first + second))
            else:
                values.add(float(first))
                if match[2]:
                    values.add(float(second))
    return values


def _value(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
    raise ValueError('Only numeric + - * / expressions are supported')


def _parse_side(text, left):
    """Longest valid numeric suffix/prefix; bounded and never executes input."""
    text = text[-220:] if left else text[:220]
    match = _RUN_LEFT.search(text) if left else _RUN_RIGHT.search(text)
    if match is None:
        return None
    run = match[0].strip()
    if not run:
        return None
    if left:
        starts = [0] + [m.start() for m in re.finditer(r'(?<=[\s(])(?=[\d(])', run)]
        fragments = (run[i:] for i in sorted(set(starts))[:24])
    else:
        ends = [len(run)] + [m.start() for m in re.finditer(r'(?<=[\d)])(?=[\s.)])', run)]
        fragments = (run[:i] for i in sorted(set(ends), reverse=True)[:24])
    for fragment in fragments:
        try:
            node = ast.parse(fragment.strip(), mode='eval').body
            if sum(1 for _ in ast.walk(node)) > 100:
                continue
            value = _value(node)
            if abs(value) > 1e12:
                continue
            return node, value
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
            continue
    return None


def _operands(node, qvalues):
    # A literal fractional quantity (e.g. 1/4) is an atomic question amount.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        value = _value(node)
        if value.denominator != 1 and _key(value) in qvalues:
            return [_key(value)]
    if isinstance(node, (ast.Constant, ast.UnaryOp)):
        return [_key(_value(node))]
    if isinstance(node, ast.BinOp):
        return _operands(node.left, qvalues) + _operands(node.right, qvalues)
    raise ValueError('Unsupported expression')


def _features(question, text):
    qvalues = _numbers(question, words=True)
    qden = max(1, len(qvalues))
    normalized = _normalize(text)
    # value -> (question ancestors, ungrounded numeric ancestors, depth)
    state = {v: ({v}, set(), 0) for v in qvalues}
    counts = {v: 0 for v in qvalues}
    used_q, used_derived, computed, unknown_operands = set(), set(), set(), set()
    all_operands = set()
    verified = parsed = maxdepth = 0
    for match in list(re.finditer('=', normalized))[:120]:
        left = _parse_side(normalized[:match.start()], True)
        right = _parse_side(normalized[match.end():], False)
        if left is None or right is None or not isinstance(left[0], ast.BinOp):
            continue
        parsed += 1
        if not math.isclose(float(left[1]), float(right[1]), rel_tol=1e-5, abs_tol=1e-6):
            continue
        verified += 1
        operands = _operands(left[0], qvalues)
        roots, unknown, depth = set(), set(), 0
        for operand in operands:
            all_operands.add(operand)
            if operand in qvalues:
                used_q.add(operand)
                counts[operand] += 1
                roots.add(operand)
            if operand in computed:
                used_derived.add(operand)
            if operand in state:
                prev_roots, prev_unknown, prev_depth = state[operand]
                roots.update(prev_roots)
                unknown.update(prev_unknown)
                depth = max(depth, prev_depth)
            else:
                unknown.add(operand)
                unknown_operands.add(operand)
        result = _key(right[1])
        # Repeated equal-valued results retain alternative known derivations.
        if result in computed:
            roots.update(state[result][0])
            unknown.update(state[result][1])
        state[result] = roots, unknown, depth + 1
        maxdepth = max(maxdepth, depth + 1)
        computed.add(result)
    answer = parse_answer(text)
    # Computed nodes are bounded to 1e12; larger parsed answers cannot have
    # a trace and need not be converted to float (which may overflow).
    exact_answer = Fraction(answer) if answer is not None else None
    akey = _key(exact_answer) if exact_answer is not None and abs(exact_answer) <= 1e12 else None
    trace = state[akey] if akey in computed else (set(), set(), 0)
    final_q, final_unknown, final_depth = trace
    mentioned = _numbers(text, words=True) & qvalues
    return (
        math.log1p(len(qvalues)), math.log1p(verified), verified / max(1, parsed),
        len(used_q) / qden, len(final_q) / qden,
        len(used_q - final_q) / qden, len(mentioned - final_q) / qden,
        float(akey in computed), math.log1p(final_depth), math.log1p(maxdepth),
        len(final_unknown) / max(1, len(final_q) + len(final_unknown)),
        len(unknown_operands) / max(1, len(all_operands)), math.log1p(len(final_q)),
        len(computed - used_derived - ({akey} if akey is not None else set())) / max(1, len(computed)),
        len(final_q & used_q) / max(1, len(used_q)),
        sum(c > 1 for c in counts.values()) / qden,
        math.log1p(parsed - verified), float(answer is not None),
    )


def extract(bank):
    """Return [N,4,D] float32, preserving input row and candidate order.

    Only question and candidate text are read. IDs, gold labels, embeddings and
    candidate code are not accessed. Missing equations produce availability zeros.
    """
    result = np.empty((len(bank), 4, DIM), dtype=np.float32)
    for i, row in enumerate(bank):
        if len(row['candidates']) != 4:
            raise ValueError('Expected exactly four candidates per question')
        for j, candidate in enumerate(row['candidates']):
            result[i, j] = _features(row['question'], candidate['text'])
    return result
