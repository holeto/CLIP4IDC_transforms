"""Re-split a converted TRANCE set so that val and test hold unseen transformations.

A transformation is its grammar program (trance_util.program_key). The text
key would not do: it keeps the direction of in/out moves, which the program
drops, so a split by text leaks in/out programs into val/test.

Every distinct program goes, with all of its generation groups (trans_ids),
to exactly one split. Within each kind (the kinds of its steps: color,
material, shape, size, inner, in, out) a seeded random --test_frac of the
programs goes to test and --val_frac to val, so every kind keeps its share of
the groups. A held-out program is a new (descriptor, action) combination, not
a new action: the script fails unless every action and every descriptor of the
set still occurs in train.

Only data.jsonl and split.json are new: the rows are reordered (train, val,
test; file order within a split) and keep idx, trans_id and member. The
source's properties.json and values.json are copied and image/ links to the
source's images, so nothing is re-rendered or converted. data.h5 and caption
references are not carried over (the references follow the split; make them
with preprocess/trance_gt.py if needed). The result is checked by loading it
with the program key and --test_transformation_generalization.

Usage:
    python preprocess/trance_heldout_split.py <src_dir> <dst_dir> \
        [--test_frac 0.1] [--val_frac 0.05] [--seed 0]
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.trance_util import load_trance_split, program_key  # noqa: E402

SPLITS = ("train", "val", "test")
COPIED_FILES = ("properties.json", "values.json")


def step_kind(action):
    """Kind of a grammar action: its attribute, or inner/in/out for moves."""
    parts = action.split(".")
    if parts[0] != "position":
        return parts[0]
    return parts[1] if parts[1] in ("in", "out") else "inner"


def program_kind(program):
    return tuple(step_kind(action) for _, action in program)


def read_groups(src):
    # The raw lines of data.jsonl and, per trans_id, its program and lines.
    # Lines stay unparsed text, so the copy needs no second read and little
    # memory. Returns (lines, {trans_id: program}, {trans_id: [line index]}).
    lines, program_of, lines_of = [], {}, defaultdict(list)
    with open(os.path.join(src, "data.jsonl")) as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            program = program_key(row["transformations"])
            trans_id = row["trans_id"]
            # Members of a group must share the program, or the group could
            # not go to one split as one transformation.
            assert program_of.setdefault(trans_id, program) == program, \
                "members of trans_id {} have different programs".format(trans_id)
            lines_of[trans_id].append(len(lines))
            lines.append(line if line.endswith("\n") else line + "\n")
    return lines, program_of, lines_of


def assign_programs(programs, test_frac, val_frac, seed):
    # Split of every program: per kind, a seeded shuffle of the programs in
    # a fixed order, the first ones to test, the next to val. Returns
    # {program: split name}.
    by_kind = defaultdict(list)
    for program in programs:
        by_kind[program_kind(program)].append(program)
    rng = random.Random(seed)
    split_of = {}
    for kind in sorted(by_kind):
        kind_programs = sorted(by_kind[kind])
        rng.shuffle(kind_programs)
        n_test = int(round(test_frac * len(kind_programs)))
        n_val = int(round(val_frac * len(kind_programs)))
        for i, program in enumerate(kind_programs):
            split_of[program] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split_of


def check_train_coverage(split_of):
    # Actions are never held out as a whole: every action and descriptor of
    # the set must occur in some train program. Returns the missing ones as
    # (actions, descriptors), both empty if train covers the set.
    seen = {"action": set(), "desc": set()}
    used = {"action": set(), "desc": set()}
    for program, name in split_of.items():
        for desc, action in program:
            used["action"].add(action)
            used["desc"].add(desc)
            if name == "train":
                seen["action"].add(action)
                seen["desc"].add(desc)
    return sorted(used["action"] - seen["action"]), sorted(used["desc"] - seen["desc"])


def summarize(split_of, program_of, lines_of):
    # Programs, groups and samples per split and per kind, as
    # {split: {kind text or "all": {"programs", "groups", "samples"}}}.
    summary = {name: defaultdict(Counter) for name in SPLITS}
    for program, name in split_of.items():
        for kind in ("+".join(program_kind(program)), "all"):
            summary[name][kind]["programs"] += 1
    for trans_id, program in program_of.items():
        name = split_of[program]
        for kind in ("+".join(program_kind(program)), "all"):
            summary[name][kind]["groups"] += 1
            summary[name][kind]["samples"] += len(lines_of[trans_id])
    return {name: {kind: dict(c) for kind, c in sorted(kinds.items())} for name, kinds in summary.items()}


def write_split(src, dst, lines, split_of, program_of, lines_of, args):
    os.makedirs(dst)
    order = {name: [] for name in SPLITS}
    for trans_id, program in program_of.items():
        order[split_of[program]].extend(lines_of[trans_id])
    rows = [sorted(order[name]) for name in SPLITS]

    tmp = os.path.join(dst, "data.jsonl.tmp")
    with open(tmp, "w") as fp:
        for indices in rows:
            fp.writelines(lines[i] for i in indices)
    os.replace(tmp, os.path.join(dst, "data.jsonl"))

    n_groups = Counter(split_of[p] for p in program_of.values())
    n_programs = Counter(split_of.values())
    split = {
        "split_name": list(SPLITS),
        "n_samples": [len(indices) for indices in rows],
        "n_transformations": [n_groups[name] for name in SPLITS],
        "n_programs": [n_programs[name] for name in SPLITS],
        "heldout": {"source": os.path.abspath(src), "key": "program", "test_frac": args.test_frac,
                    "val_frac": args.val_frac, "seed": args.seed},
    }
    with open(os.path.join(dst, "split.json"), "w") as fp:
        json.dump(split, fp, indent=2)

    for name in COPIED_FILES:
        if os.path.isfile(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
    image_dir = os.path.join(os.path.abspath(src), "image")
    if os.path.isdir(image_dir):
        try:
            os.symlink(image_dir, os.path.join(dst, "image"))
        except OSError as e:
            print("could not link image/ ({}); use --features_path {}".format(e, image_dir))
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="Converted TRANCE set (data.jsonl, split.json, image/).")
    parser.add_argument("dst", help="New directory for the re-split set (must not exist).")
    parser.add_argument("--test_frac", type=float, default=0.10, help="Share of the programs of each kind in test.")
    parser.add_argument("--val_frac", type=float, default=0.05, help="Share of the programs of each kind in val.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_check", action="store_true", help="Skip loading the result with the loader's checks.")
    args = parser.parse_args()
    assert not os.path.exists(args.dst), "{} exists".format(args.dst)
    assert 0 <= args.test_frac and 0 <= args.val_frac and args.test_frac + args.val_frac < 1

    lines, program_of, lines_of = read_groups(args.src)
    split_of = assign_programs(set(program_of.values()), args.test_frac, args.val_frac, args.seed)
    missing_actions, missing_descs = check_train_coverage(split_of)
    if missing_actions or missing_descs:
        sys.exit("train would miss actions {} and descriptors {}; try another --seed or smaller fractions".format(
            missing_actions, ["_".join(d) for d in missing_descs]))

    split = write_split(args.src, args.dst, lines, split_of, program_of, lines_of, args)
    summary = summarize(split_of, program_of, lines_of)
    with open(os.path.join(args.dst, "heldout_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    for name in SPLITS:
        print("{:5s} {}".format(name, "  ".join("{} {programs}/{groups}/{samples}".format(kind, **c)
                                                 for kind, c in summary[name].items())))
    print("(programs/groups/samples per kind)")

    # The loader's own checks: no program and no initial scene in two splits.
    if not args.no_check:
        for name in SPLITS:
            if split["n_samples"][SPLITS.index(name)]:
                load_trance_split(args.dst, name, key="program", test_transformation_generalization=True)
                break
        print("checked: load_trance_split accepts it with the program key and transformation generalization")


if __name__ == "__main__":
    main()
