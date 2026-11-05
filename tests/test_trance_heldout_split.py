"""CPU checks of the program key and of preprocess/trance_heldout_split.py.

Run from the repo root: `python tests/test_trance_heldout_split.py` (or with
pytest). Needs neither torch nor the grammar package, data or a GPU.
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "preprocess"))

from dataloaders.trance_util import load_trance_split, program_key, transformation_key  # noqa: E402
import trance_heldout_split as heldout  # noqa: E402

TOOL = os.path.join(ROOT, "preprocess", "trance_heldout_split.py")
DESCS = ["small_red_metal_cube", "large_blue_rubber_sphere", "medium_gray_rubber_cylinder",
         "small_cyan_metal_sphere"]


def edit(obj_key, attr, val):
    return {"obj_idx": 0, "attr": attr, "val": val, "options": [val], "obj_key": obj_key}


def move(obj_key, direction, n, kind):
    return {"obj_idx": 0, "attr": "position", "val": [direction, n], "options": [[direction, n]],
            "obj_key": obj_key, "type": kind}


def write_set(path, groups, n_samples, n_members=2):
    # groups: [steps] in data.jsonl order, trans_id = position. Every row
    # gets its own initial scene, so only programs can span splits.
    rows = []
    for trans_id, steps in enumerate(groups):
        for member in range(n_members):
            idx = len(rows)
            images = {"Camera_Center": "img_{}.png".format(idx)}
            rows.append({"idx": idx, "trans_id": trans_id, "member": member,
                         "states": [{"images": images, "objects": [{"color": "red", "position": [idx, 0]}]},
                                    {"images": images}],
                         "transformations": steps})
    with open(os.path.join(path, "data.jsonl"), "w") as fp:
        fp.write("\n".join(json.dumps(row) for row in rows) + "\n")
    with open(os.path.join(path, "split.json"), "w") as fp:
        json.dump({"split_name": ["train", "val", "test"], "n_samples": n_samples}, fp)
    with open(os.path.join(path, "properties.json"), "w") as fp:
        json.dump({"marker": 1}, fp)


def full_set():
    # Every descriptor with 8 actions, the in/out moves in two directions,
    # so in/out programs repeat with different texts.
    groups = []
    for desc in DESCS:
        groups += [[edit(desc, "color", "green")], [edit(desc, "color", "yellow")],
                   [edit(desc, "material", "glass")],
                   [move(desc, "front", 1, "inner")], [move(desc, "left", 2, "inner")],
                   [move(desc, "front", 1, "in")], [move(desc, "behind", 2, "in")],
                   [move(desc, "right", 1, "out")], [move(desc, "left", 2, "out")]]
    return groups


def test_program_key_drops_in_out_direction():
    a = [move(DESCS[0], "front", 1, "in")]
    b = [move(DESCS[0], "behind", 2, "in")]
    c = [move(DESCS[0], "front", 1, "inner")]
    assert program_key(a) == program_key(b) == (((("small", "red", "metal", "cube"), "position.in"),))
    assert transformation_key(a) != transformation_key(b)
    assert program_key(c) == ((("small", "red", "metal", "cube"), "position.front.1"),)


def test_loader_key_program():
    with tempfile.TemporaryDirectory() as path:
        # The same in-program in train and test, with two directions.
        write_set(path, [[move(DESCS[0], "front", 1, "in")], [edit(DESCS[0], "color", "blue")],
                         [move(DESCS[0], "behind", 2, "in")]], [2, 2, 2])
        test_text = load_trance_split(path, "test", test_transformation_generalization=True)
        assert [s["target_id"] for s in test_text] == [2, 2]
        assert [s["target_id"] for s in load_trance_split(path, "test", key="program")] == [0, 0]
        try:
            load_trance_split(path, "test", key="program", test_transformation_generalization=True)
        except AssertionError as e:
            assert "spans splits" in str(e)
            return
        raise AssertionError("an in-program in train and test was accepted with the program key")


def test_heldout_split():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        os.makedirs(os.path.join(src, "image"))
        groups = full_set()
        write_set(src, groups, [2 * len(groups), 0, 0])
        out = subprocess.run([sys.executable, TOOL, src, dst, "--test_frac", "0.25", "--val_frac", "0.25"],
                             check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
        assert "checked" in out, out

        with open(os.path.join(dst, "split.json")) as fp:
            split = json.load(fp)
        split_of_row = [n for n, k in zip(split["split_name"], split["n_samples"]) for _ in range(k)]
        with open(os.path.join(dst, "data.jsonl")) as fp:
            rows = [json.loads(line) for line in fp]
        assert sorted(r["idx"] for r in rows) == list(range(2 * len(groups)))
        splits_of_program, splits_of_group = defaultdict(set), defaultdict(set)
        for row, name in zip(rows, split_of_row):
            splits_of_program[program_key(row["transformations"])].add(name)
            splits_of_group[row["trans_id"]].add(name)
        assert all(len(s) == 1 for s in splits_of_program.values())
        assert all(len(s) == 1 for s in splits_of_group.values())
        # 28 programs (in/out merge two directions each): per kind a quarter
        # in test and a quarter in val, one in- and one out-program (two
        # groups each) among them.
        assert len(splits_of_program) == 28 and split["n_programs"] == [14, 7, 7], split
        assert split["n_transformations"] == [18, 9, 9], split

        for name in ("val", "test"):
            assert all(load_trance_split(dst, name, key="program"))
        assert os.path.isfile(os.path.join(dst, "properties.json"))
        assert os.path.islink(os.path.join(dst, "image"))


def test_coverage_check():
    a, b = ("small", "red", "metal", "cube"), ("large", "blue", "rubber", "sphere")
    split_of = {((a, "color.blue"),): "train", ((b, "size.small"),): "train", ((b, "color.blue"),): "test"}
    assert heldout.check_train_coverage(split_of) == ([], [])
    split_of[((b, "position.in"),)] = "test"
    assert heldout.check_train_coverage(split_of) == (["position.in"], [])


def test_assignment_is_seeded_and_stratified():
    programs = set(program_key(steps) for steps in full_set())
    first = heldout.assign_programs(programs, 0.25, 0.25, seed=3)
    assert first == heldout.assign_programs(programs, 0.25, 0.25, seed=3)
    per_kind = defaultdict(lambda: defaultdict(int))
    for program, name in first.items():
        per_kind[heldout.program_kind(program)][name] += 1
    for kind, counts in per_kind.items():
        assert counts["test"] == counts["val"] == sum(counts.values()) // 4, (kind, counts)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print("{} tests passed".format(len(tests)))
