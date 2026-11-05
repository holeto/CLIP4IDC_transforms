"""Stage-2 grammar metrics split by whether the transformation occurs in train.

Under scene generalization (the default setup) a val/test transformation may
also occur in the train split, on other scenes. For the decodes of every epoch
of a run (hyp_val_ep_<epoch>.json, hyp_ep_<epoch>.json, written by
main_task_caption.eval_epoch_grammar) this recomputes the metrics of that
function on three subsets of groups: seen (the transformation occurs in
train), unseen (it does not) and all. "all" must equal the run's
metrics.jsonl, which is checked when the file is there.

A transformation is identified by its grammar program (trance_util.program_key),
as the grammar loader identifies it. Until 2026-11-04 this script used the text
key, which keeps the direction of in/out moves: an in/out test transformation
counted as unseen when train had the same program with another direction (on
trance_center_1step, 78 of the 133 text-unseen test groups). Reads only the
decodes and the data set (no model, no GPU).

Usage:
    python split_metrics_by_seen.py --run_dir <stage-2 output dir> \
        --data_path <data dir> --output_dir <dir> [--splits val test]
"""
import argparse
import glob
import json
import logging
import os
import re

from dataloaders.trance_util import load_trance_split, program_key
from modules import trance_execution, trance_grammar

logger = logging.getLogger("logger")

SUBSETS = ("seen", "unseen", "all")
# Columns of the summary table, as metric names of eval_epoch_grammar.
TABLE = ("pair_explains_group", "group_every_explains_all", "exact")


def hyp_path(run_dir, split, epoch):
    # The naming of main_task_caption.hyp_file.
    return os.path.join(run_dir, "hyp_{}ep_{}.json".format("" if split == "test" else split + "_", epoch))


def epochs_of(run_dir, splits):
    """Epochs whose decodes exist for every split, in order."""
    found = None
    for split in splits:
        pattern = os.path.basename(hyp_path(run_dir, split, "*"))
        regex = re.compile("^" + re.escape(pattern).replace(r"\*", r"(\d+)") + "$")
        epochs = {int(m.group(1)) for m in (regex.match(os.path.basename(p))
                                            for p in glob.glob(hyp_path(run_dir, split, "*"))) if m}
        found = epochs if found is None else found & epochs
    return sorted(found or [])


def load_split(data_path, split, train_keys):
    """Samples of an eval split for trance_execution.execution_metrics, and
    the subset of every sample: {sample_id: (trans_id, initial objects,
    final objects)}, {sample_id: "seen" or "unseen"}."""
    samples, subset = {}, {}
    for s in load_trance_split(data_path, split):
        samples[s["sample_id"]] = (s["trans_id"], s["objects"], s["final_objects"])
        subset[s["sample_id"]] = "seen" if program_key(s["transformations"]) in train_keys else "unseen"
    return samples, subset


def subset_metrics(rows, samples, ids, geometry):
    """Metrics of eval_epoch_grammar on the samples ids (whole groups): the
    match rates in % and the execution metrics without the verdicts."""
    ids = set(ids)
    rows = [r for r in rows if r["image_id"] in ids]
    if not rows:
        return None
    # As in eval_epoch_grammar: a complete decode is the program of its rules.
    programs = {r["image_id"]: trance_grammar.program_of_derivation(r["rules"]) if r["complete"] else None
                for r in rows}
    metrics = {key: 100.0 * sum(r[key] for r in rows) / len(rows) for key in ("exact", "valid", "complete")}
    execution = trance_execution.execution_metrics(programs, {i: samples[i] for i in ids}, geometry)
    metrics.update((key, value) for key, value in execution.items() if key != "verdicts")
    metrics["reasons"] = dict(metrics["reasons"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="Output directory of a --target grammar stage-2 run.")
    parser.add_argument("--data_path", required=True, help="The run's converted TRANCE set.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(os.path.join(args.output_dir, "log.txt"))])
    logger.info("args: %s", vars(args))
    geometry = trance_grammar.geometry_of(args.data_path)

    train_keys = {program_key(s["transformations"]) for s in load_trance_split(args.data_path, "train")}
    splits = {}
    for split in args.splits:
        samples, subset = load_split(args.data_path, split, train_keys)
        ids = {name: [i for i in samples if subset[i] == name] for name in ("seen", "unseen")}
        ids["all"] = list(samples)
        splits[split] = (samples, ids)
        for name in SUBSETS:
            groups = {samples[i][0] for i in ids[name]}
            logger.info("%s %s: %d pairs, %d groups", split, name, len(ids[name]), len(groups))

    # The run's own numbers, to check the "all" subset (last line of an epoch counts).
    logged = {}
    path = os.path.join(args.run_dir, "metrics.jsonl")
    if os.path.isfile(path):
        for line in open(path):
            row = json.loads(line)
            logged[row["epoch"]] = row

    results = []
    max_diff = 0.0
    for epoch in epochs_of(args.run_dir, args.splits):
        for split, (samples, ids) in splits.items():
            with open(hyp_path(args.run_dir, split, epoch)) as f:
                rows = json.load(f)
            entry = {"epoch": epoch, "split": split}
            for name in SUBSETS:
                entry[name] = subset_metrics(rows, samples, ids[name], geometry)
            if epoch in logged and split in logged[epoch]:
                for key in TABLE:
                    max_diff = max(max_diff, abs(entry["all"][key] - logged[epoch][split][key]))
            results.append(entry)
            logger.info("epoch %d %s: %s", epoch + 1, split, " | ".join(
                "{} {}".format(name, " ".join("{:.1f}".format(entry[name][key]) if entry[name] else "-"
                                              for key in TABLE)) for name in SUBSETS))

    with open(os.path.join(args.output_dir, "seen_unseen.jsonl"), "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    logger.info("columns per subset: %s", ", ".join(TABLE))
    if logged:
        logger.info("largest difference of 'all' from metrics.jsonl: %.6f", max_diff)


if __name__ == "__main__":
    main()
