"""Reading TRANCE (converted layout: data.jsonl + split.json + image/) for CLIP4IDC.

The target of a TRANCE sample is a JSON transformation, but the decoder of CLIP4IDC is a text decoder.
`transformation_to_text` is the single place where the transformation is turned into text; replace it
when the target stops being a string.

Multi-sample sets (several (before, after) pairs per transformation, see TVR_multi_sample) use the same
layout: `idx` is a unique running id per sample, `trans_id` is shared by all members of a transformation,
and `split.json` splits whole transformations, so the splits are cut by `n_samples`.

`trans_id` is the generator's group index, not the content: if the same transformation is drawn twice,
the two groups get different `trans_id`s. A key identifies the content instead, and every sample gets a
`target_id` (one per distinct key, shared across splits), which the contrastive loss and the group-level
retrieval metric use. The key follows the target: `transformation_key` (the text) for the text target,
`program_key` (the grammar program) for the grammar target. They differ for in/out moves, whose text
keeps the direction while the program does not, so two texts can be one program.

Two evaluation setups, chosen with --test_transformation_generalization:
- scene generalization (default): a transformation may appear in several splits, on different scenes;
  the loader logs how many transformations of val/test also occur in train.
- transformation generalization (flag set): no transformation (by the key) may appear in two splits.
In both, no initial scene may appear in two splits.
"""
import json
import logging
import os
from collections import defaultdict

logger = logging.getLogger("logger")


def add_trance_args(parser):
    parser.add_argument("--test_transformation_generalization", action="store_true",
                        help="TRANCE: reject a data set whose val/test transformations also occur in train "
                             "(default: allow it and log the share, i.e. test generalization to new scenes).")


def transformation_to_text(transformations):
    """Placeholder text target. Uses only the member-invariant fields (obj_key, attr, val, type)."""
    steps = []
    for trans in transformations:
        obj = trans["obj_key"].replace("_", " ")
        if trans["attr"] == "position":
            direction, step = trans["val"]
            steps.append(f"{obj} position {direction} {step} {trans['type']}")
        else:
            steps.append(f"{obj} {trans['attr']} {trans['val']}")
    return "; ".join(steps)


def transformation_key(transformations):
    """Identity of a transformation: samples with the same key have the same target.

    It must follow the target, so it is the text for now. The steps are already in the generator's
    canonical order (since the 2026-10-26 grammar constraints), so the same transformation drawn twice
    gives the same key; older sets may hold one transformation in two step orders as two keys.
    """
    return transformation_to_text(transformations)


def transformation_program(transformations):
    """Grammar program [(descriptor, action)] of the steps of a converted data.jsonl row.

    obj_key spells the stage-0 descriptor as size_color_material_shape, the order of the grammar's R.
    Actions are written as the grammar writes them: attr.value, position.<direction>.<n> for inner
    moves, position.in / position.out (no direction) for in- and out-moves.
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


def program_key(transformations):
    """Identity of a transformation for the grammar target: its program as a tuple.

    The steps are in the generator's canonical order, so equal programs give equal keys.
    """
    return tuple(transformation_program(transformations))


# Keys selectable in load_trance_split, by the target they belong to.
KEYS = {"text": transformation_key, "program": program_key}


def load_trance_split(data_path, subset, initial_camera="Camera_Center", final_camera="Camera_Center",
                      test_transformation_generalization=False, key="text"):
    """Samples of one split as a list of dicts, in the order of data.jsonl.

    Each sample: sample_id (str of the unique idx), trans_id, member, target_id, text, bef_image,
    aft_image (file names inside the image directory), transformations (the row's steps), objects
    (its stage-0 objects) and final_objects (those of its last state). target_id numbers the distinct
    keys of the whole file in order of first appearance, so it is the same in every split; `key` names
    the key function in KEYS ("text" or "program", see the module docstring). With
    test_transformation_generalization, a transformation (by that key) in two splits is an error;
    otherwise the share of the split's transformations that also occur in train is logged (for val and
    test).
    """
    key_of = KEYS[key]
    with open(os.path.join(data_path, "data.jsonl"), "r") as fp:
        rows = [json.loads(line) for line in fp if line.strip()]
    with open(os.path.join(data_path, "split.json"), "r") as fp:
        split = json.load(fp)

    assert sum(split["n_samples"]) == len(rows), \
        "split.json n_samples sums to {}, data.jsonl has {} rows".format(sum(split["n_samples"]), len(rows))
    assert subset in split["split_name"], "unknown split {}".format(subset)

    split_of_row = []
    for name, n in zip(split["split_name"], split["n_samples"]):
        split_of_row += [name] * n

    splits_of_trans, texts_of_trans = defaultdict(set), defaultdict(set)
    splits_of_key, trans_of_key = defaultdict(set), defaultdict(set)
    splits_of_scene = defaultdict(set)
    target_id_of_key = {}
    samples = []
    for row, name in zip(rows, split_of_row):
        text = transformation_to_text(row["transformations"])
        key = key_of(row["transformations"])
        target_id = target_id_of_key.setdefault(key, len(target_id_of_key))
        splits_of_trans[row["trans_id"]].add(name)
        texts_of_trans[row["trans_id"]].add(text)
        splits_of_key[key].add(name)
        trans_of_key[key].add(row["trans_id"])
        # A scene is its initial objects (attributes and positions); rows without them are not checked.
        objects = row["states"][0].get("objects")
        if objects:
            splits_of_scene[tuple(sorted(json.dumps(o, sort_keys=True) for o in objects))].add(name)
        if name != subset:
            continue
        samples.append({
            "sample_id": str(row["idx"]),
            "trans_id": row["trans_id"],
            "member": row["member"],
            "target_id": target_id,
            "text": text,
            "bef_image": row["states"][0]["images"][initial_camera],
            "aft_image": row["states"][-1]["images"][final_camera],
            # read by the grammar target (dataloader_trance_grammar.py)
            "transformations": row["transformations"],
            "objects": row["states"][0].get("objects"),
            # read by the execution check (modules/trance_execution.py)
            "final_objects": row["states"][-1].get("objects"),
        })

    for trans_id, names in splits_of_trans.items():
        assert len(names) == 1, "transformation {} spans splits {}".format(trans_id, sorted(names))
    for trans_id, texts in texts_of_trans.items():
        assert len(texts) == 1, "members of transformation {} have different texts: {}".format(trans_id, texts)
    for scene, names in splits_of_scene.items():
        assert len(names) == 1, "an initial scene appears in splits {}".format(sorted(names))
    assert len({s["sample_id"] for s in samples}) == len(samples), "idx is not unique"

    # The same transformation drawn twice gets two trans_ids, which the converter may put in different splits.
    if test_transformation_generalization:
        for key, names in splits_of_key.items():
            assert len(names) == 1, \
                "transformation {!r} (trans_ids {}) spans splits {}; split by transformation, not by trans_id".format(
                    key, sorted(trans_of_key[key]), sorted(names))
    elif subset != "train" and samples:
        keys = {key_of(s["transformations"]) for s in samples}
        shared = sum("train" in splits_of_key[key] for key in keys)
        logger.info("%.0f%% of transformations in the %s set (%d of %d) are contained in the train set "
                    "(key: %s)", 100. * shared / len(keys), subset, shared, len(keys), key)

    return samples
