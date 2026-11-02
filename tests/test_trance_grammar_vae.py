"""CPU checks of the Grammar VAE and the TRANCE grammar target.

Run from the repo root: `python tests/test_trance_grammar_vae.py` (or with pytest). No GPU, data or CLIP weight
needed; the TRANCE grammar is found as in modules/trance_grammar.py, and its fixture of generated
transformations (grammar/tests/fixtures/generated.json, 30 transformations x 4 members) is the test data.
"""
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import trance_grammar as tg  # noqa: E402
from modules.module_gvae import GrammarVAE, kl_loss, masked_greedy_decode, reconstruction_loss  # noqa: E402

GEOMETRY = tg.grammar.Geometry()


def _groups():
    path = os.path.join(os.path.dirname(tg.grammar.__file__), "tests", "fixtures", "generated.json")
    with open(path) as f:
        groups = json.load(f)
    for group in groups:
        group["program"] = [(tuple(desc), action) for desc, action in group["programs"][0]]
    return groups


GROUPS = _groups()


def _transformations(program):
    # Converted data.jsonl steps of a program (the inverse of program_of_transformations).
    steps = []
    for desc, action in program:
        parts = action.split(".")
        step = {"obj_key": "_".join(desc), "attr": parts[0]}
        if parts[0] != "position":
            step["val"] = parts[1]
        elif parts[1] in ("in", "out"):
            step.update(val=["front", 1], type=parts[1])
        else:
            step.update(val=[parts[1], int(parts[2])], type="inner")
        steps.append(step)
    return steps


def test_program_of_transformations():
    transformations = [
        {"obj_key": "small_yellow_glass_cube", "attr": "color", "val": "cyan", "options": ["cyan"]},
        {"obj_key": "small_yellow_glass_cube", "attr": "position", "val": ["behind", 2], "type": "inner"},
        {"obj_key": "large_red_metal_sphere", "attr": "position", "val": ["left", 1], "type": "out"},
        {"obj_key": "medium_gray_rubber_cylinder", "attr": "position", "val": ["front", 2], "type": "in"},
    ]
    program = tg.program_of_transformations(transformations)
    assert program == [
        (("small", "yellow", "glass", "cube"), "color.cyan"),
        (("small", "yellow", "glass", "cube"), "position.behind.2"),
        (("large", "red", "metal", "sphere"), "position.out"),
        (("medium", "gray", "rubber", "cylinder"), "position.in"),
    ]
    rules = tg.derivation_of(program)
    assert len(rules) == tg.MAX_LEN and rules[-1] == tg.NOOP
    assert tg.program_of_derivation(rules) == program
    for group in GROUPS:
        assert tg.program_of_transformations(_transformations(group["program"])) == group["program"]


def test_teacher_masks():
    # The true rule is allowed at every step, and init_desc <= program <= cfg.
    for group in GROUPS:
        rules = tg.derivation_of(group["program"])
        masks = {mode: tg.teacher_masks(mode, rules, group["members"], GEOMETRY) for mode in tg.MASKS}
        assert all(m[np.arange(tg.MAX_LEN), rules].all() for m in masks.values())
        assert not (masks["init_desc"] & ~masks["program"]).any()
        assert not (masks["program"] & ~masks["cfg"]).any()
        # Padding allows only the no-op.
        n = sum(r != tg.NOOP for r in rules)
        assert (masks["init_desc"][n:].sum(1) == 1).all() and masks["init_desc"][n:, tg.NOOP].all()


def test_teacher_masks_reject_repeated_edit():
    group = GROUPS[0]
    desc, action = group["program"][0]
    program = [(desc, action), (desc, action.split(".")[0] + ".gray")]
    try:
        tg.teacher_masks("init_desc", tg.derivation_of(program), group["members"], GEOMETRY)
    except ValueError as e:
        assert "rejects" in str(e)
        return
    raise AssertionError("a second edit of one attribute was accepted")


def test_cfg_masker():
    for group in GROUPS[:5]:
        masker = tg.CFGMasker()
        for rule in tg.derivation_of(group["program"]):
            allowed = [i for i, ok in enumerate(masker.mask()) if ok]
            expected = [tg.NOOP] if rule == tg.NOOP else tg.cfg.RULES_OF[tg.cfg.RULES[rule].lhs]
            assert allowed == expected
            masker.apply(rule)
        assert masker.done


def _batch(mode, groups):
    rules = [tg.derivation_of(g["program"]) for g in groups]
    masks = [tg.teacher_masks(mode, r, g["members"], GEOMETRY) for r, g in zip(rules, groups)]
    return torch.tensor(rules), torch.tensor(np.stack(masks))


def test_gvae_loss_and_gradients():
    torch.manual_seed(0)
    model = GrammarVAE(tg.K, tg.MAX_LEN, latent_dim=8, hypers={"hidden": 16, "dense": 16})
    for mode in tg.MASKS:
        rules, masks = _batch(mode, GROUPS[:6])
        logits, z_mean, z_log_var = model(rules)
        assert logits.shape == (6, tg.MAX_LEN, tg.K) and z_mean.shape == (6, 8)
        recon, kl = reconstruction_loss(logits, rules, masks), kl_loss(z_mean, z_log_var)
        model.zero_grad()
        (recon + kl).backward()
        assert torch.isfinite(recon) and torch.isfinite(kl) and recon > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Steps whose mask allows only the true rule add nothing.
    rules, masks = _batch("init_desc", GROUPS[:2])
    only_true = torch.nn.functional.one_hot(rules, tg.K).bool()
    assert reconstruction_loss(torch.randn(2, tg.MAX_LEN, tg.K), rules, only_true) == 0


def test_masked_decode():
    # Under the init_desc mask any logits decode to a program that is valid in
    # every member; under the program mask to a complete one.
    torch.manual_seed(0)
    for sample in (False, True):
        logits = torch.randn(len(GROUPS), tg.MAX_LEN, tg.K) * 3
        for mode in ("init_desc", "program"):
            maskers = [tg.make_masker(mode, g["members"], GEOMETRY) for g in GROUPS]
            decoded = masked_greedy_decode(logits, maskers, sample=sample).tolist()
            assert all(m.done for m in maskers)
            assert all(tg.program_of_derivation(r) is not None for r in decoded)
            if mode == "init_desc":
                assert all(tg.accepts(g["members"], r, GEOMETRY) for g, r in zip(GROUPS, decoded))
    # The true derivation decodes back from logits that favour it.
    rules = torch.tensor([tg.derivation_of(g["program"]) for g in GROUPS])
    logits = torch.nn.functional.one_hot(rules, tg.K).float() * 10
    maskers = [tg.make_masker("init_desc", g["members"], GEOMETRY) for g in GROUPS]
    assert torch.equal(masked_greedy_decode(logits, maskers), rules)


def test_grammar_dataset():
    from dataloaders.dataloader_trance_grammar import TRANCE_Grammar_DataLoader

    with tempfile.TemporaryDirectory() as path:
        os.makedirs(os.path.join(path, "image"))
        Image.new("RGB", (8, 8)).save(os.path.join(path, "image", "x.png"))
        images = {"Camera_Center": "x.png"}
        rows = []
        for trans_id, group in enumerate(GROUPS[:3]):
            for member, objects in enumerate(group["members"]):
                rows.append({"idx": len(rows), "trans_id": trans_id, "member": member,
                             "states": [{"objects": objects, "images": images}, {"objects": [], "images": images}],
                             "transformations": _transformations(group["program"])})
        with open(os.path.join(path, "data.jsonl"), "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
        with open(os.path.join(path, "split.json"), "w") as f:
            json.dump({"split_name": ["train", "val", "test"], "n_samples": [4, 4, 4],
                       "n_transformations": [1, 1, 1]}, f)

        for task_type in ("retrieval", "caption"):
            dataset = TRANCE_Grammar_DataLoader("test", path, os.path.join(path, "image"), None,
                                                task_type=task_type, image_resolution=16)
            rules, rule_valid, rule_mask, bef, aft, image_mask, last = dataset[1]
            assert rules.tolist() == tg.derivation_of(GROUPS[2]["program"])
            assert rule_valid.sum() == sum(r != tg.NOOP for r in rules)
            assert rule_mask.shape == (tg.MAX_LEN, tg.K) and rule_mask[np.arange(tg.MAX_LEN), rules].all()
            assert bef.shape == (1, 3, 16, 16) and image_mask.tolist() == [1, 1]
            assert last == ("9" if task_type == "caption" else dataset.target_ids[1])
            assert dataset.members_of("9") == GROUPS[2]["members"]


if __name__ == "__main__":
    random.seed(0)
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print("{} tests passed".format(len(tests)))
