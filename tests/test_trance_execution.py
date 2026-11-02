"""CPU checks of the execution metric of the grammar target (modules/trance_execution.py).

Run from the repo root: `python tests/test_trance_execution.py` (or with pytest). No GPU or CLIP weight needed;
the TRANCE grammar is found as in modules/trance_grammar.py. The last two tests also run the oracle (true
programs must reproduce every goal) and negative checks on the generated sets ../TRANCE_dataset/trance_gvae_g4d
and trance_gvae_s1 (or $TRANCE_EXECUTION_DATA, separated by ':'), and are skipped for missing sets.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import trance_execution as te  # noqa: E402
from modules import trance_grammar as tg  # noqa: E402

GEOMETRY = tg.grammar.Geometry()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SETS = os.environ.get("TRANCE_EXECUTION_DATA", ":".join(
    os.path.join(os.path.dirname(REPO), "TRANCE_dataset", name) for name in ("trance_gvae_g4d", "trance_gvae_s1")))


def _obj(desc, position):
    size, color, material, shape = desc.split()
    return {"size": size, "color": color, "material": material, "shape": shape, "position": list(position)}


# Grid cells: the view is [-20, 20]^2, the grid [-40, 40]^2, a move step is 10.
CUBE = "small red metal cube"
SPHERE = "medium blue rubber sphere"
CYLINDER = "large green glass cylinder"
SCENE = [_obj(CUBE, (0, 0)), _obj(SPHERE, (10, 10)), _obj(CYLINDER, (30, 0))]


def _program(*steps):
    return [(tuple(desc.split()), action) for desc, action in steps]


def _final(changes, scene=SCENE):
    # The scene with {index: {attr: value}} applied.
    final = copy.deepcopy(scene)
    for idx, attrs in changes.items():
        final[idx].update(attrs)
    return final


def _check(program, changes, scene=SCENE):
    return te.check(program, scene, _final(changes, scene), GEOMETRY)


def test_attribute_steps():
    assert _check(_program((CUBE, "color.green")), {0: {"color": "green"}}) == te.OK
    assert _check(_program((CUBE, "color.green"), (CUBE, "size.large")),
                  {0: {"color": "green", "size": "large"}}) == te.OK
    # A wrong value, the right edit of the wrong object, a missing step.
    assert _check(_program((CUBE, "color.cyan")), {0: {"color": "green"}}) == te.WRONG_GOAL
    assert _check(_program((SPHERE, "color.green")), {0: {"color": "green"}}) == te.WRONG_GOAL
    assert _check(_program((CUBE, "color.green")), {0: {"color": "green", "size": "large"}}) == te.WRONG_GOAL
    # The generator never edits to the same value, nor an attribute twice.
    assert _check(_program((CUBE, "color.red")), {}) == te.INVALID_STEP
    assert _check(_program((CUBE, "color.cyan"), (CUBE, "color.green")), {0: {"color": "green"}}) == te.INVALID_STEP
    # An invisible object cannot be edited.
    assert _check(_program((CYLINDER, "color.red")), {2: {"color": "red"}}) == te.INVALID_STEP


def test_inner_moves():
    assert _check(_program((CUBE, "position.right.1")), {0: {"position": [0, 10]}}) == te.OK
    assert _check(_program((CUBE, "position.right.2")), {0: {"position": [0, 10]}}) == te.WRONG_GOAL
    # Onto the cube, and out of view (an 'out', not an inner move).
    assert _check(_program((SPHERE, "position.front-left.1")), {1: {"position": [0, 0]}}) == te.INVALID_STEP
    assert _check(_program((SPHERE, "position.behind.2")), {1: {"position": [30, 10]}}) == te.INVALID_STEP
    # One move per object.
    assert _check(_program((CUBE, "position.right.1"), (CUBE, "position.left.1")), {}) == te.INVALID_STEP


def test_in_and_out():
    # 'out' has no direction: any free destination out of view is the goal;
    # (30, 10) collides with the cylinder, (20, 20) is in view.
    out = _program((SPHERE, "position.out"))
    assert _check(out, {1: {"position": [10, 30]}}) == te.OK
    assert _check(out, {1: {"position": [30, 30]}}) == te.OK
    assert _check(out, {1: {"position": [30, 10]}}) == te.WRONG_GOAL
    assert _check(out, {1: {"position": [20, 20]}}) == te.WRONG_GOAL
    # Every move of the cube stays in view.
    assert _check(_program((CUBE, "position.out")), {0: {"position": [40, 0]}}) == te.INVALID_STEP
    # 'in': one of the free cells in view; (20, 10) collides with the sphere.
    state, reason = te.execute(_program((CYLINDER, "position.in")), SCENE, GEOMETRY)
    assert reason == te.OK and (20, 0) in state[2]["cells"] and (20, 10) not in state[2]["cells"]
    assert _check(_program((CYLINDER, "position.in")), {2: {"position": [20, 0]}}) == te.OK
    assert _check(_program((CYLINDER, "position.in")), {2: {"position": [20, 10]}}) == te.WRONG_GOAL
    # Edits after 'in' are judged at the object's new place.
    assert _check(_program((CYLINDER, "position.in"), (CYLINDER, "color.red")),
                  {2: {"position": [20, -10], "color": "red"}}) == te.OK
    assert _check(_program((CUBE, "position.in")), {}) == te.INVALID_STEP


def test_failures_are_verdicts():
    assert te.check(None, SCENE, SCENE, GEOMETRY) == te.NOT_PARSED
    assert _check(_program(("small red metal sphere", "color.green")), {}) == te.NO_OBJECT
    twins = SCENE + [_obj(CUBE, (-15, -15))]
    assert _check(_program((CUBE, "color.green")), {0: {"color": "green"}}, twins) == te.AMBIGUOUS_OBJECT
    # A final state with another number of objects.
    assert te.check(_program((CUBE, "color.green")), SCENE, _final({0: {"color": "green"}})[:2],
                    GEOMETRY) == te.WRONG_GOAL
    # The descriptor is the stage-0 one, also after the object has changed.
    assert _check(_program((CUBE, "color.green"), ("small green metal cube", "size.large")), {}) == te.NO_OBJECT


def test_execution_metrics():
    # Group 0: two members (the second scene is the first moved by one cell);
    # member a decodes the truth, member b a wrong colour. Group 1: one
    # member, unparsed decode.
    scene_b = [dict(o, position=[o["position"][0] + 1, o["position"][1]]) for o in SCENE]
    goal = {0: {"color": "green"}}
    samples = {
        "a": (0, SCENE, _final(goal)),
        "b": (0, scene_b, _final(goal, scene_b)),
        "c": (1, SCENE, _final(goal)),
    }
    programs = {"a": _program((CUBE, "color.green")), "b": _program((CUBE, "color.cyan")), "c": None}
    m = te.execution_metrics(programs, samples, GEOMETRY)
    assert m["verdicts"] == {"a": te.OK, "b": te.WRONG_GOAL, "c": te.NOT_PARSED}
    assert abs(m["sample"] - 100. / 3) < 1e-9
    assert m["group_all"] == 0
    # Pairs (a,a), (a,b) hold; (b,a), (b,b), (c,c) fail.
    assert m["n_cross_pairs"] == 5 and m["cross_pairs"] == 40
    assert m["group_any_explains_all"] == 50 and m["group_every_explains_all"] == 0
    programs["b"] = programs["a"]
    m = te.execution_metrics(programs, samples, GEOMETRY)
    assert m["group_all"] == 50 and m["group_every_explains_all"] == 50
    assert "per pair 66.7" in te.format_metrics(m)


def test_fixture_programs_execute():
    # Every program of the grammar's fixture (accepted by the init_desc mask)
    # is valid in each member alone; its own result is reproduced.
    path = os.path.join(os.path.dirname(tg.grammar.__file__), "tests", "fixtures", "generated.json")
    with open(path) as f:
        groups = json.load(f)
    n = 0
    for group in groups:
        program = [(tuple(desc), action) for desc, action in group["programs"][0]]
        for members in group["members"]:
            state, reason = te.execute(program, members, GEOMETRY)
            assert reason == te.OK, (group["trans_id"], reason)
            final = [dict({a: s[a] for a in ("size", "color", "material", "shape")}, position=list(s["cells"][0]))
                     for s in state]
            assert te.check(program, members, final, GEOMETRY) == te.OK
            n += 1
    assert n == 120


def _load(data_path):
    # Programs and samples of every row of a converted set, all splits.
    with open(os.path.join(data_path, "data.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    programs = {r["idx"]: tg.program_of_transformations(r["transformations"]) for r in rows}
    samples = {r["idx"]: (r["trans_id"], r["states"][0]["objects"], r["states"][-1]["objects"]) for r in rows}
    return programs, samples


def _perturb(program):
    # The first step with another value of its attribute, or another move.
    (desc, action), rest = program[0], program[1:]
    attr, value = action.split(".", 1)
    if attr != "position":
        values = tg.grammar.executor.ATTR_VALUES[attr]
        other = [v for v in values if v not in (value, desc[tg.grammar.executor.DESC_ATTRS.index(attr)])]
        return [(desc, "{}.{}".format(attr, other[0]))] + rest
    if value in ("in", "out"):
        return [(desc, "position." + ("out" if value == "in" else "in"))] + rest
    direction, n = value.split(".")
    return [(desc, "position.{}.{}".format(direction, "1" if n == "2" else "2"))] + rest


def test_oracle_and_negatives_on_data():
    for data_path in DATA_SETS.split(":"):
        if not os.path.isfile(os.path.join(data_path, "data.jsonl")):
            print("skip", data_path)
            continue
        geometry = tg.geometry_of(data_path)
        programs, samples = _load(data_path)
        name = os.path.basename(data_path)

        # Oracle: the true program of every sample reproduces its goal.
        m = te.execution_metrics(programs, samples, geometry)
        print("  {} oracle: {}".format(name, te.format_metrics(m)))
        assert m["reasons"] == {te.OK: len(samples)}, m["reasons"]
        assert m["cross_pairs"] == m["group_every_explains_all"] == 100

        # Wrong transformations: the program of the next group, the first
        # step perturbed, the last step dropped.
        trans_ids = sorted({s[0] for s in samples.values()})
        program_of = {samples[i][0]: programs[i] for i in samples}
        negatives = {
            "next group": {i: program_of[trans_ids[(trans_ids.index(s[0]) + 1) % len(trans_ids)]]
                           for i, s in samples.items()},
            "perturbed": {i: _perturb(p) for i, p in programs.items()},
            "step dropped": {i: p[:-1] for i, p in programs.items()},
        }
        for kind, wrong in negatives.items():
            m = te.execution_metrics(wrong, samples, geometry)
            print("  {} {}: {}".format(name, kind, te.format_metrics(m)))
            assert m["sample"] == 0 and m["cross_pairs"] == 0, (kind, m["verdicts"])


def test_grammar_loader_on_data():
    # The path of eval_epoch_grammar with --execution_oracle, without the model.
    from dataloaders.dataloader_trance_grammar import TRANCE_Grammar_DataLoader

    for data_path in DATA_SETS.split(":"):
        if not os.path.isfile(os.path.join(data_path, "data.jsonl")):
            print("skip", data_path)
            continue
        for subset in ("train", "val", "test"):
            dataset = TRANCE_Grammar_DataLoader(subset, data_path, os.path.join(data_path, "image"), None,
                                                task_type="caption", image_resolution=16)
            samples = dataset.execution_samples()
            programs = {i: tg.program_of_derivation(dataset.rules_of(i).tolist()) for i in samples}
            m = te.execution_metrics(programs, samples, dataset.geometry)
            assert m["reasons"] == {te.OK: len(dataset)}, (data_path, subset, m["reasons"])


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print("{} tests passed".format(len(tests)))
