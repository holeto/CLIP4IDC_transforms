"""CPU checks of the multi-sample TRANCE handling: masked contrastive loss, group retrieval metric, target ids.

Run from the repo root: `python tests/test_trance_multi_sample.py` (or with pytest). No GPU, data or CLIP weight needed.
"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.trance_util import load_trance_split, logger as trance_logger  # noqa: E402
from metrics import compute_metrics, group_retrieval_metrics  # noqa: E402
from modules.until_module import CrossEn  # noqa: E402


def same_target_mask(target_ids, device="cpu"):
    # CLIP4IDC._same_target_mask without building the model (eval mode, so no allgather)
    from modules.modeling import CLIP4IDC
    return CLIP4IDC._same_target_mask(types.SimpleNamespace(training=False), torch.tensor(target_ids, device=device))


def symmetric_loss(sim, target_ids):
    mask = same_target_mask(target_ids, sim.device)
    return (CrossEn()(sim, mask) + CrossEn()(sim.T, mask)) / 2


def test_mask():
    mask = same_target_mask([0, 0, 1, 2, 0])
    expected = torch.zeros(5, 5, dtype=torch.bool)
    for i, j in [(0, 1), (0, 4), (1, 4)]:
        expected[i, j] = expected[j, i] = True
    assert torch.equal(mask, expected)


def test_loss_without_duplicates_is_unchanged():
    sim = torch.randn(6, 6) * 10
    assert torch.allclose(CrossEn()(sim), CrossEn()(sim, same_target_mask(list(range(6)))))


def test_loss_equals_removing_masked_entries():
    torch.manual_seed(1)
    sim = torch.randn(5, 5) * 10
    ids = [0, 0, 1, 2, 1]
    ref = 0
    for s in (sim, sim.T):
        rows = []
        for i in range(5):
            keep = [j for j in range(5) if j == i or ids[j] != ids[i]]
            rows.append(-F.log_softmax(s[i, keep], dim=-1)[keep.index(i)])
        ref = ref + torch.stack(rows).mean()
    assert torch.allclose(symmetric_loss(sim, ids), ref / 2)


def test_loss_finite_gradients_and_zero_for_one_transformation():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for dtype in (torch.float32, torch.float16):
            for ids in ([0, 1, 2, 3], [0, 0, 1, 2], [5, 5, 5, 5]):
                torch.manual_seed(0)
                sim = (torch.randn(4, 4) * 30).to(device, dtype).requires_grad_()
                loss = symmetric_loss(sim, ids)
                loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(sim.grad).all(), (device, dtype, ids)
                if len(set(ids)) == 1:
                    # no negatives at all: the loss is exactly 0, still part of the graph
                    assert float(loss) == 0. and loss.requires_grad and float(sim.grad.abs().sum()) == 0.


def test_group_metric_equals_per_sample_with_one_member():
    rng = np.random.default_rng(0)
    sim = rng.normal(size=(30, 30))
    b2t, t2b = group_retrieval_metrics(sim, np.arange(30), np.arange(30))
    t2v, v2t = compute_metrics(sim), compute_metrics(sim.T)
    for key in ("R1", "R5", "R10", "MR", "MeanR"):
        assert np.isclose(t2b[key], t2v[key]) and np.isclose(b2t[key], v2t[key]), key


def test_group_metric_not_capped_by_members():
    # 5 transformations x 4 members; the model separates transformations but not members
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(5), 4)
    sim = 10. * (ids[:, None] == ids[None, :]) + rng.uniform(size=(20, 20))
    assert compute_metrics(sim)["R1"] < 50          # per-sample: capped near 1/4
    b2t, t2b = group_retrieval_metrics(sim, ids, ids)
    assert b2t["R1"] == 100. and t2b["R1"] == 100. and b2t["n"] == 5 and t2b["n"] == 5


def test_group_metric_transformation_in_two_groups():
    # transformation 0 was generated twice (groups 0 and 1), so K=3 texts and G=4 groups
    target_ids = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    group_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    sim = 10. * (target_ids[:, None] == target_ids[None, :])
    sim[:, 2:4] = 0.                          # group 1 matches nothing, group 0 still matches text 0
    sim[0, 2:4] = -1.
    b2t, t2b = group_retrieval_metrics(sim, target_ids, group_ids)
    assert (b2t["n"], t2b["n"]) == (4, 3)
    assert t2b["R1"] == 100.                  # the best group of each text counts, the other is not wrong
    assert b2t["R1"] == 75.                   # group 1 ranks its text last of 3
    assert b2t["MeanR"] == (1 + 3 + 1 + 1) / 4


def test_group_metric_rejects_mixed_group():
    try:
        group_retrieval_metrics(np.zeros((2, 2)), [0, 1], [7, 7])
    except AssertionError:
        return
    raise AssertionError("a group with two target ids was accepted")


def _write_set(path, groups, n_samples, objects_of=None):
    """groups: list of (trans_id, colour, n_members) in data.jsonl order; n_samples per split train/val/test.

    objects_of(idx), if given, gives the initial objects of a row (otherwise rows have none).
    """
    rows = []
    for trans_id, colour, n_members in groups:
        for member in range(n_members):
            idx = len(rows)
            images = {"Camera_Center": "img_{}.png".format(idx)}
            rows.append({
                "idx": idx, "trans_id": trans_id, "member": member,
                "states": [{"images": images, **({"objects": objects_of(idx)} if objects_of else {})},
                           {"images": images}],
                "transformations": [{"obj_idx": member, "attr": "color", "val": colour, "options": [colour],
                                     "obj_key": "small_red_metal_cube"}],
            })
    with open(os.path.join(path, "data.jsonl"), "w") as fp:
        fp.write("\n".join(json.dumps(row) for row in rows) + "\n")
    with open(os.path.join(path, "split.json"), "w") as fp:
        json.dump({"split_name": ["train", "val", "test"], "n_samples": n_samples}, fp)


def test_target_ids_shared_across_groups_and_splits():
    with tempfile.TemporaryDirectory() as path:
        # trans_ids 0 and 2 are the same transformation, both in train
        _write_set(path, [(0, "blue", 2), (2, "blue", 2), (1, "green", 2), (3, "gray", 2)], [4, 2, 2])
        train = load_trance_split(path, "train")
        assert [s["trans_id"] for s in train] == [0, 0, 2, 2]
        assert {s["target_id"] for s in train} == {0}
        assert [s["target_id"] for s in load_trance_split(path, "val")] == [1, 1]
        assert [s["target_id"] for s in load_trance_split(path, "test")] == [2, 2]


def test_transformation_in_two_splits_is_rejected():
    with tempfile.TemporaryDirectory() as path:
        _write_set(path, [(0, "blue", 2), (1, "green", 2), (2, "blue", 2)], [2, 2, 2])
        try:
            load_trance_split(path, "train", test_transformation_generalization=True)
        except AssertionError as e:
            assert "spans splits" in str(e)
            return
        raise AssertionError("a transformation in train and test was accepted")


class _Records(list):
    def write(self, message):
        self.append(message)

    def flush(self):
        pass


def test_transformation_in_two_splits_is_logged_by_default():
    import logging
    records = _Records()
    handler = logging.StreamHandler(records)
    level = trance_logger.level
    trance_logger.addHandler(handler)
    trance_logger.setLevel(logging.INFO)
    try:
        with tempfile.TemporaryDirectory() as path:
            # test: blue (also in train) and gray (not)
            _write_set(path, [(0, "blue", 2), (1, "green", 2), (2, "blue", 2), (3, "gray", 2)], [2, 2, 4])
            assert load_trance_split(path, "train") and not records
            test = load_trance_split(path, "test")
            assert [s["target_id"] for s in test] == [0, 0, 2, 2]
    finally:
        trance_logger.removeHandler(handler)
        trance_logger.setLevel(level)
    assert len(records) == 1 and "50% of transformations in the test set (1 of 2)" in records[0], records


def test_scene_in_two_splits_is_rejected():
    def objects_of(idx):
        # rows 0 (train) and 4 (test) share their initial scene
        return [{"color": "red", "position": [idx % 4, 0]}]
    with tempfile.TemporaryDirectory() as path:
        _write_set(path, [(0, "blue", 2), (1, "green", 2), (2, "gray", 2)], [2, 2, 2], objects_of)
        for flag in (False, True):
            try:
                load_trance_split(path, "train", test_transformation_generalization=flag)
            except AssertionError as e:
                assert "initial scene" in str(e)
                continue
            raise AssertionError("an initial scene in train and test was accepted")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print("{} tests passed".format(len(tests)))
