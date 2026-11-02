"""Execution check of TRANCE transformations on the scene description.

A decoded transformation is correct for a (before, after) pair if it turns
the pair's initial state into its final state, judged on the JSON objects of
data.jsonl (size, color, material, shape, grid position), not on pixels.

A program [(descriptor, action)] (trance_grammar.py) is applied step by step:
- the target is the only object whose stage-0 descriptor is the step's R;
  obj_idx is never read, because it differs between members;
- a step must be valid in the scene as the generator makes it, judged by
  grammar.executor.Member.valid (visibility, one edit per attribute, a new
  value, one move per object, grid, overlaps, in-move candidates); an 'in' or
  'out' also needs a free move of its kind;
- attribute edits set the value, inner moves shift the cell.
The grammar's 'in' and 'out' have no direction, so after them the object is
only known to be in a set of cells (Member's candidate cells after 'in', its
free invisible destinations after 'out').

The goal is reproduced if the final state has as many objects as the initial
one, and every object (matched by its index, which data.jsonl keeps between
states) has the final size, color, material and shape exactly, and its final
grid cell is the executed cell or one of the executed cells after 'in'/'out'.
Cells are integers, so there is no tolerance. Each such object is judged on
its own: two in-moved objects are not checked against each other's cells.

Every failure is a verdict, never an exception: see the REASONS below.
Uses the grammar package found by trance_grammar.py and nothing else.
"""
from collections import Counter, defaultdict

# Imported for its side effect: it finds the grammar package and puts it on
# sys.path.
from modules import trance_grammar  # noqa: F401
from grammar.executor import DESC_ATTRS, Member  # noqa: E402

OK = "ok"
# Verdicts other than OK, in the order a check can reach them.
NOT_PARSED = "not_parsed"
NO_OBJECT = "no_object"
AMBIGUOUS_OBJECT = "ambiguous_object"
INVALID_STEP = "invalid_step"
WRONG_GOAL = "wrong_goal"
REASONS = (OK, NOT_PARSED, NO_OBJECT, AMBIGUOUS_OBJECT, INVALID_STEP, WRONG_GOAL)


def execute(program, objects, geometry):
    """Applies a program to the stage-0 objects of one scene.

    Args:
        program: [(descriptor, action)] as trance_grammar.program_of_derivation
            returns it, or None for a decode that did not parse.
        objects: the stage-0 objects of data.jsonl ({size, color, material,
            shape, position}).
        geometry: grammar.Geometry of the data set.

    Returns (state, reason). state is a list with one dict per object: its
    size, color, material, shape and cells (the possible grid cells), or None
    unless reason is OK.
    """
    if program is None:
        return None, NOT_PARSED
    member = Member(objects, geometry)
    attrs = [{attr: obj[attr] for attr in DESC_ATTRS} for obj in objects]
    for desc, action in program:
        found = [i for i, o in enumerate(member.objects) if o.desc == tuple(desc)]
        if not found:
            return None, NO_OBJECT
        if len(found) > 1:
            return None, AMBIGUOUS_OBJECT
        idx = found[0]
        parts = action.split(".")
        if not member.valid(idx, action):
            return None, INVALID_STEP
        # Member.valid leaves the existence of a free in/out move to the
        # common-move test of all members, here it is one member.
        if parts[0] == "position" and parts[1] in ("in", "out") and not member.moves_for(idx, parts[1]):
            return None, INVALID_STEP
        member.apply(idx, action)
        if parts[0] != "position":
            attrs[idx][parts[0]] = parts[1]
    state = [dict(a, cells=[tuple(c) for c in o.cells]) for a, o in zip(attrs, member.objects)]
    return state, OK


def reproduces(state, final):
    """Whether an executed state matches the final objects of the sample."""
    if len(state) != len(final):
        return False
    for obj, goal in zip(state, final):
        if any(obj[attr] != goal[attr] for attr in DESC_ATTRS):
            return False
        if tuple(goal["position"]) not in obj["cells"]:
            return False
    return True


def check(program, initial, final, geometry):
    """Verdict of a program on one sample: OK or one of the failure REASONS."""
    state, reason = execute(program, initial, geometry)
    if reason != OK:
        return reason
    return OK if reproduces(state, final) else WRONG_GOAL


def execution_metrics(programs, samples, geometry):
    """Execution metrics of decoded programs over an eval split.

    Args:
        programs: {sample_id: program or None}, the decode of each sample.
        samples: {sample_id: (trans_id, initial objects, final objects)}, every
            member of every group in the split.
        geometry: grammar.Geometry of the data set.

    Returns a dict (rates in %, counts as integers):
        sample: per-sample execution accuracy, the decode of pair i turns i's
            initial state into i's final state;
        group_all: groups (trans_id) where every member's own decode does;
        cross_pairs: ordered pairs (i, j) of members of one group, i == j
            included, where the decode of i reproduces j's goal;
        group_any_explains_all / group_every_explains_all: groups where at
            least one / every member's decode reproduces the goal of all
            members;
        n_samples, n_groups, n_cross_pairs;
        reasons: Counter of the per-sample verdicts;
        verdicts: {sample_id: verdict of its own decode}.
    """
    members = defaultdict(list)
    for sample_id, (trans_id, _, _) in samples.items():
        members[trans_id].append(sample_id)

    verdicts = {}
    cross = 0
    n_cross = 0
    groups_all = groups_any = groups_every = 0
    for trans_id, ids in members.items():
        # explains[i]: the decode of member i reproduces every member's goal.
        explains = []
        for i in ids:
            ok_on = []
            for j in ids:
                verdict = check(programs.get(i), samples[j][1], samples[j][2], geometry)
                if i == j:
                    verdicts[i] = verdict
                ok_on.append(verdict == OK)
            cross += sum(ok_on)
            n_cross += len(ok_on)
            explains.append(all(ok_on))
        groups_all += all(verdicts[i] == OK for i in ids)
        groups_any += any(explains)
        groups_every += all(explains)

    def rate(k, n):
        return 100.0 * k / n if n else 0.0

    n_groups = len(members)
    return {
        "sample": rate(sum(v == OK for v in verdicts.values()), len(verdicts)),
        "group_all": rate(groups_all, n_groups),
        "cross_pairs": rate(cross, n_cross),
        "group_any_explains_all": rate(groups_any, n_groups),
        "group_every_explains_all": rate(groups_every, n_groups),
        "n_samples": len(verdicts),
        "n_groups": n_groups,
        "n_cross_pairs": n_cross,
        "reasons": Counter(verdicts.values()),
        "verdicts": verdicts,
    }


def format_metrics(metrics):
    """One log line of execution_metrics."""
    counts = ", ".join("{} {}".format(r, metrics["reasons"][r]) for r in REASONS if metrics["reasons"][r])
    return ("Execution: per pair {sample:.1f} - all members of a group {group_all:.1f} - "
            "cross-member pairs {cross_pairs:.1f} - one decode explains its group {group_any_explains_all:.1f} - "
            "every decode explains its group {group_every_explains_all:.1f} "
            "({n_samples} pairs, {n_groups} groups, {n_cross_pairs} cross pairs; {counts})").format(
                counts=counts, **metrics)

