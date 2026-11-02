"""The TRANCE transformation grammar as the Grammar VAE sees it.

The grammar and its masking rules live in the `grammar` package of
TVR_multi_sample (see its notes/multi_sample_generator_vs_grammar.md). This
module is the only place that imports it. It is looked up, in order, under
$TRANCE_GRAMMAR_ROOT, <repo>/third_party (for a vendored copy of grammar/) and
the sibling checkout ../TVR_multi_sample.

A transformation becomes the leftmost derivation of its parse tree, padded
with the no-op rule to MAX_LEN. Three masks restrict the rules per step:
- init_desc: grammar.MaskedDecoder with every member's stage-0 objects, so a
  step must be valid in each member's scene;
- program: MaskedDecoder without init_desc, judged by the program alone;
- cfg: the original GVAE mask, only the rules of the leftmost non-terminal.
"""
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAMMAR_ROOTS = [
    os.environ.get("TRANCE_GRAMMAR_ROOT"),
    os.path.join(_REPO, "third_party"),
    os.path.join(os.path.dirname(_REPO), "TVR_multi_sample"),
]


def _import_grammar():
    # The grammar package of the first root that has one.
    for root in GRAMMAR_ROOTS:
        if root and os.path.isfile(os.path.join(root, "grammar", "decoder.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            import grammar
            return grammar
    raise ImportError(
        "TRANCE grammar not found; set TRANCE_GRAMMAR_ROOT to the directory that contains grammar/ "
        "(looked in {})".format([r for r in GRAMMAR_ROOTS if r]))


grammar = _import_grammar()
from grammar import cfg  # noqa: E402
from grammar.decoder import MAX_STEPS, MaskedDecoder  # noqa: E402
from modules.grammar_args import MASKS  # noqa: E402

K = cfg.K
NOOP = cfg.NOOP
# T -> S C, then per step 9 rules at most (S, R, 4 values, A and 2 values)
# and one C rule.
MAX_LEN = 1 + 10 * MAX_STEPS


def program_of_transformations(transformations):
    """Program [(descriptor, action)] of the steps of a converted data.jsonl row.

    obj_key spells the stage-0 descriptor as size_color_material_shape, which
    is the order of R.
    """
    program = []
    for trans in transformations:
        desc = tuple(trans["obj_key"].split("_"))
        if trans["attr"] != "position":
            action = "{}.{}".format(trans["attr"], trans["val"])
        elif trans["type"] in ("in", "out"):
            action = "position." + trans["type"]
        else:
            direction, n = trans["val"]
            action = "position.{}.{}".format(direction, n)
        program.append((desc, action))
    return program


def derivation_of(program):
    """Leftmost derivation of a program, padded with the no-op to MAX_LEN."""
    rules = cfg.derivation(cfg.parse(cfg.program_tokens(program)))
    assert len(rules) <= MAX_LEN, "derivation of {} rules is longer than {}".format(len(rules), MAX_LEN)
    return rules + [NOOP] * (MAX_LEN - len(rules))


def program_of_derivation(rules):
    """Program of a (padded) derivation, or None if it is incomplete or invalid."""
    try:
        return cfg.tokens_program(cfg.derive([r for r in rules if r != NOOP]))
    except (cfg.ParseError, IndexError):
        return None


def program_text(program):
    """Readable form, e.g. small yellow glass cube color.cyan; ..."""
    return "; ".join("{} {}".format(" ".join(desc), action) for desc, action in program)


class CFGMasker:
    """The original GVAE mask: the rules whose left-hand side is the leftmost
    non-terminal, only the no-op once the derivation is complete. Same
    interface as grammar.MaskedDecoder."""

    def __init__(self):
        self.stack = ["T"]

    @property
    def done(self):
        return not self.stack

    def mask(self):
        allowed = [False] * K
        if self.done:
            allowed[NOOP] = True
        else:
            for rule in cfg.RULES_OF[self.stack[-1]]:
                allowed[rule] = True
        return allowed

    def apply(self, rule):
        if rule == NOOP:
            return
        assert cfg.RULES[rule].lhs == self.stack[-1], "rule {} does not expand {}".format(rule, self.stack[-1])
        self.stack.pop()
        self.stack.extend(s for s in reversed(cfg.RULES[rule].rhs) if not cfg.is_terminal(s))


def make_masker(mode, members, geometry):
    """A fresh masker for one sample. members: stage-0 objects of every member
    of its transformation (read only by the init_desc mask)."""
    if mode == "init_desc":
        return MaskedDecoder(members, geometry)
    if mode == "program":
        return MaskedDecoder(geometry=geometry, use_init_desc=False)
    if mode == "cfg":
        return CFGMasker()
    raise ValueError("unknown grammar mask {}".format(mode))


def teacher_masks(mode, rules, members, geometry):
    """Masks (MAX_LEN, K) of a padded true derivation, step by step.

    Raises ValueError if the mask rejects one of its rules, e.g. data from a
    generator before the grammar constraints (non-canonical step order, or an
    attribute edited twice).
    """
    masker = make_masker(mode, members, geometry)
    masks = np.zeros((len(rules), K), dtype=bool)
    for t, rule in enumerate(rules):
        masks[t] = masker.mask()
        if not masks[t, rule]:
            raise ValueError("the {} mask rejects rule {} at step {} of {}".format(
                mode, rule, t, program_text(program_of_derivation(rules) or [])))
        masker.apply(rule)
    return masks


def accepts(members, rules, geometry):
    """Whether a (padded) derivation is complete and valid in every member's scene."""
    ok, _ = grammar.accepts(members, [r for r in rules if r != NOOP], geometry)
    return ok


def geometry_of(data_path):
    """Geometry from the dataset's properties.json, the generator's otherwise."""
    path = os.path.join(data_path, "properties.json")
    if os.path.isfile(path):
        with open(path) as f:
            return grammar.Geometry(json.load(f))
    return grammar.Geometry()
