"""Probabilities of the stage-2 grammar decoder on chosen image pairs.

Tells whether the model is uncertain or confidently wrong where its greedy
decode fails. For every chosen pair, the rule logits of a CLIP4IDCGrammar
checkpoint are scored under the masked decoder exactly as decoding applies
it: log p(derivation) = sum over steps t of log masked_softmax(logits_t,
mask_t)[rule_t], with mask_t from a fresh masker advanced along that same
derivation (padding steps allow only the no-op and add 0). So the
probabilities of all derivations a mask allows sum to 1.

Pairs: the failing ones (not an exact match in the --hyp decodes of an eval
run, or in this script's own decodes without --hyp) and, as a calibration
baseline, --n_baseline successful ones drawn with --seed. Per pair and mask:
- the greedy decode (recomputed; compared with the stored rules) and the
  log-probabilities of it and of the true derivation;
- the first step where they differ: both rules' probabilities there and the
  rank of the true rule among the allowed ones;
- the weakest decision of the greedy decode (smallest probability of a chosen
  rule at a step with several allowed rules);
- the distribution over the one-step programs the mask allows at the start
  (all of them for the --focus pairs, the top ones otherwise).

Masks (--masks): group is the init_desc mask as training and eval use it,
built from the stage-0 objects of every member of the pair's group; own is
the init_desc mask of the pair's own scene only. Under the group mask the
descriptor is nearly always forced, and a step with one allowed rule gives no
gradient, so the descriptor logits that the own mask reads were never
trained.

Writes to --output_dir: pairs.json (per pair), summary.json (per mask and
status, plus the --focus pairs) and logits.npz (rule logits of the chosen
pairs, for further analysis without the image encoder).

Builds the model like main_task_caption.init_model, with the finetuning
settings of --config, and runs on CPU.

Usage:
    python decoder_probabilities.py --checkpoint <run>/pytorch_model.bin.38 \
        --data_path <data dir> --output_dir <dir> \
        --hyp val=<run>/hyp_val_ep_38.json test=<run>/hyp_ep_38.json \
        [--n_baseline 300] [--focus 33029:2]
"""
import argparse
import copy
import json
import logging
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from dataloaders.dataloader_trance_grammar import TRANCE_Grammar_DataLoader
from modules import trance_grammar
from modules.modeling_grammar import CLIP4IDCGrammar
from modules.module_gvae import masked_greedy_decode, masked_log_softmax

logger = logging.getLogger("logger")
MASK_VARIANTS = ("group", "own")
TOP_PROGRAMS = 5


def model_args(config_path):
    """Namespace of the stage-2 grammar model: --config's common and finetuning keys."""
    with open(config_path) as fp:
        config = yaml.safe_load(fp) or {}
    values = dict(config.get("common") or {})
    values.update(config.get("finetuning") or {})
    values.update(task_type="caption", target="grammar", local_rank=0)
    return argparse.Namespace(**values)


def load_model(checkpoint, args):
    state = torch.load(checkpoint, map_location="cpu")
    model = CLIP4IDCGrammar.from_pretrained(args.cross_model, args.decoder_model, state_dict=state,
                                            task_config=args)
    return model.eval()


def rule_label(rule):
    if rule == trance_grammar.NOOP:
        return "noop"
    r = trance_grammar.cfg.RULES[rule]
    return "{} -> {}".format(r.lhs, " ".join(r.rhs) or "eps")


def score(logits, rules, masker):
    # Masked log-probability of a padded derivation, step by step. Returns
    # (total, steps) with steps[t] = (allowed (K,) bool, log-probs (K,)), or
    # (None, steps) if the mask rejects a rule or the derivation stops early
    # (rule -1 of a dead-end greedy decode).
    steps = []
    total = 0.0
    for t, rule in enumerate(rules):
        if rule < 0:
            return None, steps
        allowed = torch.tensor(masker.mask(), dtype=torch.bool)
        log_p = masked_log_softmax(logits[t], allowed)
        steps.append((allowed, log_p))
        if not allowed[rule]:
            return None, steps
        total += float(log_p[rule])
        masker.apply(rule)
    return total, steps


def symbol_at(rules, t):
    return trance_grammar.cfg.RULES[rules[t]].lhs if rules[t] != trance_grammar.NOOP else "done"


def divergence(greedy, true, true_steps):
    # The first step where the greedy and the true derivation differ. Both
    # prefixes are equal up to it, so the true derivation's mask is the one
    # decoding saw. Returns None if they do not differ.
    for t, (g, r) in enumerate(zip(greedy, true)):
        if g != r:
            break
    else:
        return None
    if t >= len(true_steps):
        return {"step": t, "note": "the mask rejects the true rule before this step"}
    allowed, log_p = true_steps[t]
    p = log_p.exp()
    order = sorted((int(i) for i in allowed.nonzero().flatten()), key=lambda i: -float(p[i]))
    return {"step": t, "symbol": symbol_at(true, t),
            "rule_greedy": rule_label(g) if g >= 0 else "dead end", "rule_true": rule_label(r),
            "p_rule_greedy": float(p[g]) if g >= 0 else None, "p_rule_true": float(p[r]),
            # 1 + the allowed rules with a strictly higher probability.
            "rank_true": 1 + sum(float(p[i]) > float(p[r]) for i in order),
            "n_allowed": len(order),
            "top": [[rule_label(i), float(p[i])] for i in order[:TOP_PROGRAMS]]}


def weakest_decision(rules, steps):
    # The chosen rule with the smallest probability among the steps where the
    # mask allowed several rules; None if every step was forced.
    best = None
    for t, (allowed, log_p) in enumerate(steps):
        if int(allowed.sum()) > 1:
            p = float(log_p[rules[t]].exp())
            if best is None or p < best["p"]:
                best = {"step": t, "symbol": symbol_at(rules, t), "rule": rule_label(rules[t]), "p": p}
    return best


def program_distribution(logits, primed, true_program, greedy_program, keep_all):
    # Probabilities of every one-step program the mask allows at the start.
    # primed: a fresh masker whose first valid set is computed already; each
    # candidate is scored on a copy of it.
    programs = []
    for desc, action in sorted(primed.valid):
        program = [(desc, action)]
        log_p, _ = score(logits, trance_grammar.derivation_of(program), copy.deepcopy(primed))
        programs.append((log_p, trance_grammar.program_text(program)))
    programs.sort(key=lambda x: -x[0])
    texts = [text for _, text in programs]
    true_text = trance_grammar.program_text(true_program)
    greedy_text = None if greedy_program is None else trance_grammar.program_text(greedy_program)
    return {"n_one_step": len(programs),
            # The rest of the mass is on programs of several steps.
            "p_one_step_total": sum(math.exp(lp) for lp, _ in programs),
            "rank_true": texts.index(true_text) + 1 if true_text in texts else None,
            "rank_greedy": texts.index(greedy_text) + 1 if greedy_text in texts else None,
            "top": [[text, math.exp(lp)] for lp, text in (programs if keep_all else programs[:TOP_PROGRAMS])]}


def analyse_pair(logits, sample, group_members, true_rules, geometry, stored_rules, keep_all):
    """Per mask variant: greedy decode, probabilities, divergence and program distribution."""
    true_program = trance_grammar.program_of_transformations(sample["transformations"])
    out = {}
    for variant in MASK_VARIANTS:
        members = group_members if variant == "group" else [sample["objects"]]

        def fresh():
            return trance_grammar.make_masker("init_desc", members, geometry)

        masker = fresh()
        greedy = masked_greedy_decode(logits.unsqueeze(0), [masker])[0].tolist()
        greedy_program = trance_grammar.program_of_derivation(greedy) if masker.done else None
        log_p_greedy, greedy_steps = score(logits, greedy, fresh())
        log_p_true, true_steps = score(logits, true_rules, fresh())
        # Computes the first valid set once, for all candidate programs.
        primed = fresh()
        primed.valid_map
        record = {
            "greedy_program": None if greedy_program is None else trance_grammar.program_text(greedy_program),
            "greedy_rules": greedy,
            "exact": greedy == true_rules,
            "log_p_greedy": log_p_greedy, "log_p_true": log_p_true,
            "p_greedy": None if log_p_greedy is None else math.exp(log_p_greedy),
            "p_true": None if log_p_true is None else math.exp(log_p_true),
            "divergence": divergence(greedy, true_rules, true_steps),
            "weakest_decision": weakest_decision(greedy, greedy_steps),
            "programs": program_distribution(logits, primed, true_program, greedy_program, keep_all),
        }
        if variant == "group" and stored_rules is not None:
            record["greedy_matches_stored"] = greedy == stored_rules
        out[variant] = record
    return out


def quantiles(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    q = np.quantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    return {"n": len(values), "min": q[0], "p10": q[1], "p25": q[2], "median": q[3], "p75": q[4],
            "p90": q[5], "max": q[6], "mean": float(np.mean(values))}


def auroc(failing, success):
    """P(score of a random failing pair < that of a random successful one), ties count half."""
    if not failing or not success:
        return None
    wins = sum((f < s) + 0.5 * (f == s) for f in failing for s in success)
    return wins / (len(failing) * len(success))


def summarize(pairs):
    summary = {}
    for variant in MASK_VARIANTS:
        per_status = {}
        for status in ("failing", "baseline"):
            chosen = [p for p in pairs if p["status"] == status]
            recs = [p["masks"][variant] for p in chosen]
            if not recs:
                continue
            div = [r["divergence"] for r in recs if r["divergence"] and "rank_true" in r["divergence"]]
            weak = [r["weakest_decision"]["p"] for r in recs if r["weakest_decision"]]
            per_status[status] = {
                "n": len(recs),
                "exact": sum(r["exact"] for r in recs),
                "greedy_matches_stored": sum(r.get("greedy_matches_stored", False) for r in recs)
                if any("greedy_matches_stored" in r for r in recs) else None,
                "true_rejected_by_mask": sum(r["log_p_true"] is None for r in recs),
                "p_greedy": quantiles([r["p_greedy"] for r in recs]),
                "p_true": quantiles([r["p_true"] for r in recs]),
                "weakest_decision_p": quantiles(weak),
                "divergence_symbol": dict(Counter(d["symbol"] for d in div)),
                "divergence_p_rule_greedy": quantiles([d["p_rule_greedy"] for d in div]),
                "divergence_p_rule_true": quantiles([d["p_rule_true"] for d in div]),
                "divergence_rank_true": dict(sorted(Counter(d["rank_true"] for d in div).items())),
                "program_rank_true": dict(sorted(Counter(str(r["programs"]["rank_true"])
                                                         for r in recs).items())),
                "n_one_step_programs": quantiles([r["programs"]["n_one_step"] for r in recs]),
                "names_true_object": sum(r["greedy_program"] is not None and
                                         r["greedy_program"].split(" ")[:4] == p["true"].split(" ")[:4]
                                         for r, p in zip(recs, chosen)),
            }
        # How well the decoder's own confidence tells failures from successes.
        fail = [p["masks"][variant] for p in pairs if p["status"] == "failing"]
        succ = [p["masks"][variant] for p in pairs if p["status"] == "baseline"]
        per_status["auroc_p_greedy"] = auroc([r["p_greedy"] or 0 for r in fail], [r["p_greedy"] or 0 for r in succ])
        per_status["auroc_weakest_decision"] = auroc(
            [r["weakest_decision"]["p"] if r["weakest_decision"] else 1 for r in fail],
            [r["weakest_decision"]["p"] if r["weakest_decision"] else 1 for r in succ])
        summary[variant] = per_status
    return summary


def parse_hyp(items):
    hyp = {}
    for item in items or []:
        split, path = item.split("=", 1)
        hyp[split] = path
    return hyp


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="pytorch_model.bin.<epoch> of a stage-2 grammar run")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features_path", default=None, help="image directory (default <data_path>/image)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/default_one_step.yaml",
                        help="settings of the model (common and finetuning sections)")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--hyp", nargs="*", help="split=path of the decodes of an eval run (hyp_*ep_<epoch>.json); "
                        "without one, the split's pairs are all decoded here to find the failing ones")
    parser.add_argument("--n_baseline", type=int, default=300, help="successful pairs drawn as the baseline")
    parser.add_argument("--focus", nargs="*", default=[], help="trans_id:member pairs reported in full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cross_model", default="cross-base")
    parser.add_argument("--decoder_model", default="decoder-base")
    cli = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    os.makedirs(cli.output_dir, exist_ok=True)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    start = time.time()

    args = model_args(cli.config)
    args.cross_model, args.decoder_model = cli.cross_model, cli.decoder_model
    assert args.grammar_mask == "init_desc", "this script scores the init_desc mask only"
    model = load_model(cli.checkpoint, args)
    features_path = cli.features_path or os.path.join(cli.data_path, "image")
    hyp_paths = parse_hyp(cli.hyp)
    focus = {tuple(int(x) for x in f.split(":")) for f in cli.focus}

    # Rule logits of a list of dataset indices, in that order.
    def logits_of(dataset, indices):
        loader = DataLoader(Subset(dataset, indices), batch_size=cli.batch_size, num_workers=cli.num_workers)
        out = []
        with torch.no_grad():
            for batch in loader:
                _, _, _, bef_image, aft_image, image_mask = batch[:-1]
                out.append(model.rule_logits(bef_image, aft_image, image_mask).float())
        return torch.cat(out) if out else torch.zeros(0)

    # Choose the pairs: failing and successful by the stored decodes, or by
    # decoding every pair of the split here.
    datasets, stored, candidates = {}, {}, []
    for split in cli.splits:
        # The group scope keeps one set of teacher masks per group; this script
        # builds both mask variants itself from group_members_of.
        dataset = TRANCE_Grammar_DataLoader(split, cli.data_path, features_path, tokenizer=None,
                                            task_type="caption", grammar_mask="init_desc", init_desc_scope="group")
        datasets[split] = dataset
        index_of = {s["sample_id"]: i for i, s in enumerate(dataset.samples)}
        if split in hyp_paths:
            for record in json.load(open(hyp_paths[split])):
                stored[(split, record["image_id"])] = record["rules"]
                candidates.append((split, index_of[record["image_id"]], not record["exact"]))
        else:
            logger.info("%s: no --hyp, decoding all %d pairs to find the failing ones", split, len(dataset))
            indices = list(range(len(dataset)))
            logits = logits_of(dataset, indices)
            for i, row in zip(indices, logits):
                sample = dataset.samples[i]
                masker = trance_grammar.make_masker("init_desc", dataset.group_members_of(sample["sample_id"]),
                                                    dataset.geometry)
                rules = masked_greedy_decode(row.unsqueeze(0), [masker])[0].tolist()
                candidates.append((split, i, rules != dataset.rules_of(sample["sample_id"]).tolist()))
    failing = [(s, i) for s, i, fail in candidates if fail]
    success = sorted((s, i) for s, i, fail in candidates if not fail)
    baseline = random.Random(cli.seed).sample(success, min(cli.n_baseline, len(success)))
    chosen = [(s, i, "failing") for s, i in failing] + [(s, i, "baseline") for s, i in sorted(baseline)]

    def is_focus(split, i):
        return (datasets[split].samples[i]["trans_id"], datasets[split].samples[i]["member"]) in focus

    taken = set(failing) | set(baseline)
    chosen += [(s, i, "focus") for s, i, _ in candidates if is_focus(s, i) and (s, i) not in taken]
    logger.info("pairs: %d failing, %d baseline of %d successful, %d focus only", len(failing), len(baseline),
                len(success), sum(st == "focus" for *_, st in chosen))

    pairs, all_logits, ids = [], [], []
    for split in cli.splits:
        dataset = datasets[split]
        mine = [(i, status) for s, i, status in chosen if s == split]
        logits = logits_of(dataset, [i for i, _ in mine])
        for (i, status), row in zip(mine, logits):
            sample = dataset.samples[i]
            sid = sample["sample_id"]
            pair = {"split": split, "sample_id": sid, "trans_id": sample["trans_id"], "member": sample["member"],
                    "status": status,
                    "true": trance_grammar.program_text(
                        trance_grammar.program_of_transformations(sample["transformations"]))}
            pair["masks"] = analyse_pair(row, sample, dataset.group_members_of(sid), dataset.rules_of(sid).tolist(),
                                         dataset.geometry, stored.get((split, sid)), is_focus(split, i))
            pairs.append(pair)
            all_logits.append(row.numpy())
            ids.append("{}:{}".format(split, sid))
        logger.info("%s: %d pairs analysed (%.0f s)", split, len(mine), time.time() - start)

    summary = summarize(pairs)
    summary["focus"] = [p for p in pairs if (p["trans_id"], p["member"]) in focus]
    summary["settings"] = dict(vars(cli), n_failing=len(failing), n_success=len(success))
    json.dump(pairs, open(os.path.join(cli.output_dir, "pairs.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(cli.output_dir, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(cli.output_dir, "logits.npz"), ids=np.array(ids),
                        logits=np.stack(all_logits) if all_logits else np.zeros(0))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("focus",)}, indent=1))

    # One line per failing or focus pair, for reading the log.
    for p in pairs:
        if p["status"] == "baseline":
            continue
        line = ["{} {}:{} {} [{}]".format(p["split"], p["trans_id"], p["member"], p["sample_id"], p["status"]),
                "  true: " + p["true"]]
        for variant in MASK_VARIANTS:
            r = p["masks"][variant]
            d = r["divergence"] or {}
            line.append("  {}: decode {!r} p={:.3g}, p(true)={}, diverges at {} ({} vs {}: {} vs {}, true rank {}/{}), "
                        "true program rank {}/{}".format(
                            variant, r["greedy_program"], r["p_greedy"] or 0,
                            "{:.3g}".format(r["p_true"]) if r["p_true"] is not None else "rejected",
                            d.get("symbol"), d.get("rule_greedy"), d.get("rule_true"),
                            "{:.3g}".format(d["p_rule_greedy"]) if d.get("p_rule_greedy") is not None else None,
                            "{:.3g}".format(d["p_rule_true"]) if d.get("p_rule_true") is not None else None,
                            d.get("rank_true"), d.get("n_allowed"),
                            r["programs"]["rank_true"], r["programs"]["n_one_step"]))
            if (p["trans_id"], p["member"]) in focus:
                line.append("    one-step programs: " + "; ".join(
                    "{} {:.3g}".format(t, q) for t, q in r["programs"]["top"]))
        print("\n".join(line))
    logger.info("done in %.0f s", time.time() - start)


if __name__ == "__main__":
    main()
