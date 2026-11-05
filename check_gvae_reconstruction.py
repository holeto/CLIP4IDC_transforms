"""Reconstruction check of the Grammar VAE of a grammar-target checkpoint.

For every group (trans_id) of a TRANCE split, the true derivation of its
transformation is encoded, its z_mean decoded greedily under a rule mask, and
the decoded program executed on every member of the group
(modules/trance_execution.py). With --init_desc_scope own (the default) the
init_desc mask reads one member's scene, so every member gets its own decode;
with group it reads all of them and the group one decode. The main numbers
are the share of pairs whose decode reproduces the goal of every member of
their group, and of groups where every member's decode does. Only the GVAE weights of the
checkpoint are loaded (no CLIP, no images), so it runs on CPU.

It tells whether the GVAE learned a usable latent space before image pairs are
trained to decode from it (stage 2). Signs of a collapsed latent:
- the decodes do not reproduce their groups, or many targets decode to the
  same program;
- the control, which decodes every group from the z of a group with another
  transformation, scores about as well (then the mask, not z, does the work).
The usual signs, a KL near 0 and few active dimensions, do not apply with
epsilon_std < 1: training samples z from N(mu, (epsilon_std * sigma)^2), while
the KL term is computed for N(mu, sigma^2), so the KL pins sigma to 1 and only
shrinks mu, whatever the codes carry (notes/GVAE_preliminary.md).

Measurements of the latent space, besides the KL and the active dimensions:
- the distribution of z_mean (per dimension, over the groups) next to the
  scale the decoder normalises z by (its first BatchNorm's running std);
- per distinct transformation, the distance from its z_mean to the nearest
  z_mean of another transformation, in raw units and in the decoder's
  normalised units;
- decodes from noisy codes, z = z_mean + s * sigma * eps for each
  --noise_scales s (training uses s = epsilon_std = 0.01, the posterior that
  the KL term describes is s = 1), and from the prior, z ~ N(0, I).

Decoding z_mean is deterministic: no reparameterisation noise, greedy
decoding. The control's donor groups and the noisy and prior codes are drawn
from generators seeded with --seed and the split.

Masks: init_desc reads the members' scenes, so it can steer a weak decode
towards a valid program by itself; cfg allows every syntactically possible
rule. The GVAE is trained under one mask (init_desc in
configs/default_one_step.yaml) and the logits of the rules it forbids get no
gradient, so a cfg decode can fail on those without a collapse.

Usage:
    python check_gvae_reconstruction.py --checkpoint <run>/pytorch_model.bin.11 \
        --data_path <data dir> --output_dir <dir> [--splits test] [--masks init_desc cfg]
"""
import argparse
import json
import logging
import os
import random
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch

from dataloaders.trance_util import load_trance_split
from modules import trance_execution, trance_grammar
from modules.grammar_args import INIT_DESC_SCOPES, MASKS
from modules.module_gvae import GrammarVAE, masked_greedy_decode

logger = logging.getLogger("logger")

# Posterior collapse diagnostic of Burda et al. (2016): a latent dimension is
# active if z_mean varies over the data by more than this.
ACTIVE_VARIANCE = 0.01


def load_gvae(path):
    """The GVAE of a CLIP4IDCGrammar checkpoint (pytorch_model.bin.<epoch>), in eval mode."""
    state = torch.load(path, map_location="cpu")
    state = {k[len("gvae."):]: v for k, v in state.items() if k.startswith("gvae.")}
    if not state:
        raise ValueError("{} has no GVAE weights; is it a --target grammar checkpoint?".format(path))
    # The sizes come from the checkpoint, so they cannot disagree with it.
    latent_dim = state["decoder.latent_input.weight"].shape[0]
    hidden = state["decoder.out.weight"].shape[1]
    gvae = GrammarVAE(trance_grammar.K, trance_grammar.MAX_LEN, latent_dim, hypers={"hidden": hidden})
    gvae.load_state_dict(state)
    return gvae.eval()


def groups_of(samples):
    """Groups of a split, in order of first appearance.

    Returns a list of dicts: trans_id, target (the program as text), rules (the
    padded true derivation) and members, [(sample_id, initial objects, final
    objects)] in member order.
    """
    by_trans = defaultdict(list)
    for sample in samples:
        by_trans[sample["trans_id"]].append(sample)
    groups = []
    for trans_id, members in by_trans.items():
        members = sorted(members, key=lambda s: s["member"])
        program = trance_grammar.program_of_transformations(members[0]["transformations"])
        groups.append({
            "trans_id": trans_id,
            "target": trance_grammar.program_text(program),
            "rules": trance_grammar.derivation_of(program),
            "members": [(s["sample_id"], s["objects"], s["final_objects"]) for s in members],
        })
    return groups


def control_donors(groups, rng):
    # For every group, a random group with another transformation, whose z it
    # is decoded from in the control. Returns None if the split has a single
    # transformation.
    targets = [g["target"] for g in groups]
    if len(set(targets)) < 2:
        return None
    donors = rng.integers(len(groups), size=len(groups))
    for i in range(len(groups)):
        while targets[donors[i]] == targets[i]:
            donors[i] = rng.integers(len(groups))
    return donors


def decode_groups(gvae, z, groups, mask, geometry, chunk, scope):
    """Greedy decodes of z (one row per group), one per member: under the
    member's own scene (scope own) or under all the group's scenes (group,
    then every member gets the same decode).

    Returns per group a list of (rules, program) in member order, program None
    if the mask did not finish the derivation or it does not parse.
    """
    decodes = []
    for start in range(0, len(groups), chunk):
        part = groups[start:start + chunk]
        with torch.no_grad():
            logits = gvae.decode(z[start:start + chunk])
        if scope == "group":
            maskers = [trance_grammar.make_masker(mask, [m[1] for m in g["members"]], geometry) for g in part]
            sizes = [1] * len(part)
        else:
            maskers = [trance_grammar.make_masker(mask, [m[1]], geometry) for g in part for m in g["members"]]
            sizes = [len(g["members"]) for g in part]
            logits = torch.repeat_interleave(logits, torch.tensor(sizes), dim=0)
        flat = [(rules, trance_grammar.program_of_derivation(rules) if masker.done else None)
                for rules, masker in zip(masked_greedy_decode(logits, maskers).tolist(), maskers)]
        k = 0
        for g, size in zip(part, sizes):
            decodes.append(flat[k:k + size] * (len(g["members"]) if size == 1 else 1))
            k += size
    return decodes


def score(groups, decodes, geometry):
    """Metrics of one decode per member (decode_groups).

    Returns (metrics, verdicts). metrics in % of pairs: exact, complete,
    members (the pair's decode reproduces its own goal) and
    pair_explains_group (the goal of every member of its group); in % of
    groups: explains_group (every member's decode reproduces every member's
    goal; with the group scope, the group's one decode does); the distinct
    decoded programs and the counts of the pairs' verdicts on their own scene.
    verdicts[g][i][j]: verdict of member i's decode on member j of group g.
    """
    complete = exact = members_ok = pairs_explain = explains = n_pairs = 0
    reasons = Counter()
    programs = set()
    verdicts = []
    for group, group_decodes in zip(groups, decodes):
        # Members often share a decode, so each program is checked once per member.
        cache = {}
        group_verdicts = []
        for rules, program in group_decodes:
            key = tuple(rules)
            if key not in cache:
                cache[key] = [trance_execution.check(program, initial, final, geometry)
                              for _, initial, final in group["members"]]
            group_verdicts.append(cache[key])
        verdicts.append(group_verdicts)
        for i, (rules, program) in enumerate(group_decodes):
            ok = [v == trance_execution.OK for v in group_verdicts[i]]
            complete += program is not None
            exact += rules == group["rules"]
            members_ok += ok[i]
            pairs_explain += all(ok)
            reasons[group_verdicts[i][i]] += 1
            if program is not None:
                programs.add(trance_grammar.program_text(program))
        n_pairs += len(group_decodes)
        explains += all(v == trance_execution.OK for row in group_verdicts for v in row)
    metrics = {
        "explains_group": 100.0 * explains / len(groups),
        "pair_explains_group": 100.0 * pairs_explain / n_pairs,
        "members": 100.0 * members_ok / n_pairs,
        "exact": 100.0 * exact / n_pairs,
        "complete": 100.0 * complete / n_pairs,
        "distinct_programs": len(programs),
        "reasons": dict(reasons),
    }
    return metrics, verdicts


def quantiles(x):
    """Summary of a 1-d tensor: min, 1%, 5%, median, 95%, max."""
    x = x.double()
    points = {"min": 0.0, "q01": 0.01, "q05": 0.05, "median": 0.5, "q95": 0.95, "max": 1.0}
    return {name: float(torch.quantile(x, q)) for name, q in points.items()}


def nearest_other(codes):
    # Distance from every code to the nearest other code; None for a single
    # code.
    if len(codes) < 2:
        return None
    distances = torch.cdist(codes.double(), codes.double())
    distances.fill_diagonal_(float("inf"))
    return distances.min(1).values


def decoder_scale(gvae):
    """Per dimension, the std the decoder divides z by (BatchNorm in eval mode)."""
    norm = gvae.decoder.norm
    return (norm.running_var + norm.eps).sqrt()


def latent_stats(gvae, z_mean, z_log_var, groups):
    """Statistics of the codes of a split.

    Returns a dict: KL to N(0, I) per dimension (mean over groups and
    dimensions, as in the loss), active dimensions (Burda et al.; an absolute
    threshold, so it depends on the scale of z_mean), the posterior std
    (mean, and its range over the groups and dimensions), z_mean's RMS and
    the quantiles of its per-dimension std over the groups, the decoder's
    normalising std per dimension, and the quantiles of the distance from
    each distinct transformation's z_mean to the nearest one of another
    transformation (nn_distance raw, nn_distance_normalised divided by the
    decoder's std per dimension).
    """
    std = (z_log_var / 2).exp()
    kl = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    scale = decoder_scale(gvae)
    # The same derivation always gets the same code, so one per transformation.
    first = {}
    for i, group in enumerate(groups):
        first.setdefault(group["target"], i)
    codes = z_mean[list(first.values())]
    nearest = nearest_other(codes)
    nearest_normalised = nearest_other(codes / scale)
    return {
        "latent_dim": z_mean.shape[1],
        "kl_per_dim": float(kl.mean()),
        "active_dims": int((z_mean.var(0) > ACTIVE_VARIANCE).sum()) if len(z_mean) > 1 else None,
        "posterior_std": float(std.mean()),
        "posterior_std_range": [float(std.min()), float(std.max())],
        "z_mean_rms": float(z_mean.pow(2).mean().sqrt()),
        "z_mean_dim_std": quantiles(z_mean.std(0)) if len(z_mean) > 1 else None,
        "decoder_std": quantiles(scale),
        "nn_distance": None if nearest is None else quantiles(nearest),
        "nn_distance_normalised": None if nearest is None else quantiles(nearest_normalised),
    }


def latent_variants(args, split, gvae, z_mean, z_log_var, donors):
    """The codes decoded for every group, as [(name, z)]: decode (z_mean),
    control (z_mean of the donor group), noise_<s> (z_mean + s * sigma * eps)
    and prior (N(0, I)). Also returns, per variant other than decode, the mean
    distance of z from z_mean, raw and in the decoder's normalised units."""
    seed = (args.seed * 1000003 + zlib.crc32(split.encode())) % 2 ** 63
    generator = torch.Generator().manual_seed(seed)
    variants = [("decode", z_mean)]
    if donors is not None:
        variants.append(("control", z_mean[donors]))
    std = (z_log_var / 2).exp()
    for s in args.noise_scales:
        eps = torch.randn(z_mean.shape, generator=generator)
        variants.append(("noise_{:g}".format(s), z_mean + s * std * eps))
    if args.prior:
        variants.append(("prior", torch.randn(z_mean.shape, generator=generator)))
    scale = decoder_scale(gvae)
    shifts = {name: {"shift": float((z - z_mean).norm(dim=1).mean()),
                     "shift_normalised": float(((z - z_mean) / scale).norm(dim=1).mean())}
              for name, z in variants[1:]}
    return variants, shifts


def format_metrics(m):
    counts = ", ".join("{} {}".format(r, m["reasons"][r]) for r in trance_execution.REASONS if r in m["reasons"])
    return ("pairs explaining their group {pair_explains_group:.1f} - every member explains its group "
            "{explains_group:.1f} - members {members:.1f} - exact {exact:.1f} - "
            "complete {complete:.1f} - {distinct_programs} distinct programs ({counts})").format(counts=counts, **m)


def check_split(args, gvae, split, geometry):
    # The control's draws depend only on --seed and the split, not on the
    # other splits checked.
    rng = np.random.default_rng([args.seed, zlib.crc32(split.encode())])
    groups = groups_of(load_trance_split(args.data_path, split))
    if args.limit:
        groups = groups[:args.limit]
    rules = torch.tensor([g["rules"] for g in groups])
    with torch.no_grad():
        z_mean, z_log_var = zip(*(gvae.encode(rules[i:i + args.chunk]) for i in range(0, len(rules), args.chunk)))
    z_mean, z_log_var = torch.cat(z_mean), torch.cat(z_log_var)
    donors = control_donors(groups, rng)
    variants, shifts = latent_variants(args, split, gvae, z_mean, z_log_var, donors)

    result = {"split": split, "groups": len(groups), "distinct_targets": len({g["target"] for g in groups}),
              "latent": latent_stats(gvae, z_mean, z_log_var, groups), "shifts": shifts, "masks": {}}
    logger.info("%s: %d groups, %d distinct targets", split, result["groups"], result["distinct_targets"])
    for key, value in result["latent"].items():
        logger.info("%s latent %s: %s", split, key, json.dumps(value))
    for name, shift in shifts.items():
        logger.info("%s %s: mean distance from z_mean %.4g, normalised %.4g", split, name, shift["shift"],
                    shift["shift_normalised"])
    per_group = [{"trans_id": g["trans_id"], "target": g["target"], "members": [m[0] for m in g["members"]]}
                 for g in groups]
    for mask in args.masks:
        result["masks"][mask] = {}
        for name, z in variants:
            start = time.time()
            decodes = decode_groups(gvae, z, groups, mask, geometry, args.chunk, args.init_desc_scope)
            metrics, verdicts = score(groups, decodes, geometry)
            result["masks"][mask][name] = metrics
            logger.info("%s, %s mask, %s: %s (%.0f s)", split, mask, name, format_metrics(metrics),
                        time.time() - start)
            if name == "decode":
                for entry, group_decodes, group_verdicts in zip(per_group, decodes, verdicts):
                    entry[mask] = {"programs": [None if program is None else trance_grammar.program_text(program)
                                                for _, program in group_decodes],
                                   "verdicts": group_verdicts}

    path = os.path.join(args.output_dir, "gvae_reconstruction_{}.json".format(split))
    with open(path, "w") as f:
        json.dump(dict(result, per_group=per_group), f, indent=1)
    logger.info("written %s", path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="pytorch_model.bin.<epoch> of a --target grammar run.")
    parser.add_argument("--data_path", required=True, help="Converted TRANCE set (data.jsonl, split.json).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--masks", nargs="+", default=["init_desc", "cfg"], choices=MASKS)
    parser.add_argument("--init_desc_scope", default="own", choices=INIT_DESC_SCOPES,
                        help="init_desc mask of each member's own scene (default; one decode per member) or of "
                             "all the group's scenes (one decode per group, as runs before 2026-11-03).")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the control's donors and the noisy codes.")
    parser.add_argument("--noise_scales", nargs="*", type=float, default=[0.01, 0.1, 1.0],
                        help="Decode z_mean + s * sigma * eps for each s (none: skip).")
    parser.add_argument("--no_prior", dest="prior", action="store_false", help="Skip the decodes from N(0, I).")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N groups of each split (0 = all).")
    parser.add_argument("--chunk", type=int, default=1024, help="Groups decoded at once.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(os.path.join(args.output_dir, "log.txt"))])
    logger.info("args: %s", vars(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gvae = load_gvae(args.checkpoint)
    geometry = trance_grammar.geometry_of(args.data_path)
    results = [check_split(args, gvae, split, geometry) for split in args.splits]
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
