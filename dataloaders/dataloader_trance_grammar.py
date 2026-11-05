"""TRANCE with the grammar target: the transformation as a GVAE derivation.

Every sample gives its derivation (rule indices, padded with the no-op to
MAX_LEN) and the teacher masks of --grammar_mask: the allowed rules at each
step of the true derivation. The init_desc mask reads the stage-0 objects of
the scenes chosen by --init_desc_scope: the sample's own scene (own, the
default), or every member's scene of its transformation (group, as the
generator requires of the whole group). The group scope leaks the other
members' scenes: on one-step TRANCE it leaves a single allowed object in
every val/test group (notes/found_failures.md).

The own scope needs one set of masks per sample, which is slow to compute
(~20 ms each), so they are computed in parallel and can be cached on disk
(--grammar_mask_cache), keyed by the data file, split, mask, scope and grammar
source.

Items keep the three text slots of the text loaders, so that
main_task_retrieval.py runs unchanged (input_ids, input_mask, segment_ids):
    rules (T,), rule_valid (T,) (1 on the derivation, 0 on the padding),
    rule_mask (T, K), bef_image, aft_image, image_mask,
then target_id for retrieval or sample_id for captioning.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import hashlib
import json
import logging
import multiprocessing
import os
from collections import defaultdict

import numpy as np

from dataloaders.dataloader_trance_retrieval import TRANCE_DataLoader
from modules import trance_grammar

logger = logging.getLogger("logger")

# Per worker process: the mask mode and geometry of the masks it computes.
_worker_setup = {}


def _init_worker(mode, geometry):
    _worker_setup.update(mode=mode, geometry=geometry)


def _packed_masks(job):
    # Teacher masks of one (key, rules, scenes) job, bit-packed along the
    # rules. Returns (key, packed masks or None, error message or None).
    key, rules, scenes = job
    try:
        masks = trance_grammar.teacher_masks(_worker_setup["mode"], rules, scenes, _worker_setup["geometry"])
    except ValueError as e:
        return key, None, str(e)
    return key, np.packbits(masks, axis=-1), None



class TRANCE_Grammar_DataLoader(TRANCE_DataLoader):
    def __init__(self, subset, data_path, features_path, tokenizer, task_type="retrieval", grammar_mask="init_desc",
                 init_desc_scope="own", mask_cache_dir=None, mask_workers=0, **kwargs):
        # A transformation is its program here: in/out moves in two directions are one target, so
        # target_id, the "spans splits" check and the seen share all go by the program.
        super(TRANCE_Grammar_DataLoader, self).__init__(subset, data_path, features_path, tokenizer, key="program",
                                                        **kwargs)
        self.task_type = task_type
        self.grammar_mask = grammar_mask
        self.init_desc_scope = init_desc_scope
        self.geometry = trance_grammar.geometry_of(data_path)

        members = defaultdict(list)
        for sample in sorted(self.samples, key=lambda s: s["member"]):
            members[sample["trans_id"]].append(sample["objects"])
        self.sample_of = {sample["sample_id"]: sample for sample in self.samples}
        self.trans_id_of = {sample["sample_id"]: sample["trans_id"] for sample in self.samples}
        # Members share the program, so the derivation is computed once per
        # transformation. The masks are too, unless they read the own scene.
        self.per_sample = grammar_mask == "init_desc" and init_desc_scope == "own"
        self.groups = {}
        jobs, sample_of_key = [], {}
        for sample in self.samples:
            trans_id = sample["trans_id"]
            if trans_id not in self.groups:
                rules = trance_grammar.derivation_of(
                    trance_grammar.program_of_transformations(sample["transformations"]))
                self.groups[trans_id] = {"rules": np.array(rules, dtype=np.int64), "members": members[trans_id]}
            key = sample["sample_id"] if self.per_sample else trans_id
            if key not in sample_of_key:
                sample_of_key[key] = sample
                jobs.append((key, self.groups[trans_id]["rules"].tolist(), self.members_of(sample["sample_id"])))
        # Bit-packed (T, ceil(K / 8)) masks per key, unpacked in __getitem__.
        self.masks = self._load_masks(data_path, subset, jobs, sample_of_key, mask_cache_dir, mask_workers)

    def _load_masks(self, data_path, subset, jobs, sample_of_key, cache_dir, workers):
        # Teacher masks of every job, from the cache file if it matches,
        # otherwise computed (in parallel) and, with a cache_dir, saved.
        path = None
        if cache_dir:
            data_file = os.path.join(data_path, "data.jsonl")
            stat = os.stat(data_file)
            identity = json.dumps([os.path.realpath(data_file), stat.st_size, stat.st_mtime_ns, subset,
                                   self.grammar_mask, self.init_desc_scope, trance_grammar.grammar_version(),
                                   trance_grammar.MAX_LEN, trance_grammar.K,
                                   repr(sorted(self.geometry.__dict__.items()))])
            path = os.path.join(cache_dir, "teacher_masks_{}_{}_{}_{}.npz".format(
                subset, self.grammar_mask, self.init_desc_scope, hashlib.md5(identity.encode()).hexdigest()[:12]))
            if os.path.isfile(path):
                cached = np.load(path)
                keys = [self._key_type(k) for k in cached["keys"].tolist()]
                if set(keys) == set(sample_of_key):
                    logger.info("teacher masks of %s (%d) from %s", subset, len(keys), path)
                    return dict(zip(keys, cached["packed"]))
                logger.info("%s does not match the samples, recomputing", path)

        workers = workers or len(os.sched_getaffinity(0))
        logger.info("computing %d teacher masks of %s (%s mask, %s scope) with %d processes",
                    len(jobs), subset, self.grammar_mask, self.init_desc_scope, workers)
        if workers > 1 and len(jobs) > 1:
            with multiprocessing.get_context("fork").Pool(workers, _init_worker,
                                                           (self.grammar_mask, self.geometry)) as pool:
                results = pool.map(_packed_masks, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
        else:
            _init_worker(self.grammar_mask, self.geometry)
            results = [_packed_masks(job) for job in jobs]
        masks = {}
        for key, packed, error in results:
            if error is not None:
                sample = sample_of_key[key]
                raise ValueError("sample {} (trans_id {}): {}".format(sample["sample_id"], sample["trans_id"], error))
            masks[key] = packed

        if path:
            os.makedirs(cache_dir, exist_ok=True)
            keys = list(masks)
            # Written under a temporary name, so that concurrent loads never
            # read a partial file.
            tmp = "{}.{}.tmp.npz".format(path[:-len(".npz")], os.getpid())
            np.savez(tmp, keys=np.array([str(k) for k in keys]), packed=np.stack([masks[k] for k in keys]))
            os.replace(tmp, path)
            logger.info("saved teacher masks of %s to %s", subset, path)
        return masks

    def _key_type(self, key):
        # Cache keys are stored as text: sample ids are text, trans_ids ints.
        return key if self.per_sample else int(key)

    def members_of(self, sample_id):
        """Stage-0 objects of the scenes the sample's init_desc mask reads
        (--init_desc_scope): [its own] or those of every member."""
        if self.init_desc_scope == "own":
            return [self.sample_of[sample_id]["objects"]]
        return self.group_members_of(sample_id)

    def group_members_of(self, sample_id):
        """Stage-0 objects of every member of the sample's transformation."""
        return self.groups[self.trans_id_of[sample_id]]["members"]

    def rules_of(self, sample_id):
        return self.groups[self.trans_id_of[sample_id]]["rules"]

    def execution_samples(self):
        """{sample_id: (trans_id, initial objects, final objects)}, the input of
        trance_execution.execution_metrics."""
        return {s["sample_id"]: (s["trans_id"], s["objects"], s["final_objects"]) for s in self.samples}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        group = self.groups[sample["trans_id"]]
        rules = group["rules"]
        packed = self.masks[sample["sample_id"] if self.per_sample else sample["trans_id"]]
        rule_mask = np.unpackbits(packed, axis=-1, count=trance_grammar.K).astype(bool)
        rule_valid = (rules != trance_grammar.NOOP).astype(np.int64)
        bef_image = self._get_rawimage(os.path.join(self.features_path, sample["bef_image"]))
        aft_image = self._get_rawimage(os.path.join(self.features_path, sample["aft_image"]))
        image_mask = np.ones(2, dtype=np.int64)
        last = np.int64(sample["target_id"]) if self.task_type == "retrieval" else sample["sample_id"]
        return rules, rule_valid, rule_mask, bef_image, aft_image, image_mask, last
