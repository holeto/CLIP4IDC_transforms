"""TRANCE with the grammar target: the transformation as a GVAE derivation.

Every sample gives its derivation (rule indices, padded with the no-op to
MAX_LEN) and the teacher masks of --grammar_mask: the allowed rules at each
step of the true derivation. The init_desc mask reads the stage-0 objects of
every member of the sample's transformation, as the grammar's decoder does.

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

import os
from collections import defaultdict

import numpy as np

from dataloaders.dataloader_trance_retrieval import TRANCE_DataLoader
from modules import trance_grammar


class TRANCE_Grammar_DataLoader(TRANCE_DataLoader):
    def __init__(self, subset, data_path, features_path, tokenizer, task_type="retrieval", grammar_mask="init_desc",
                 **kwargs):
        super(TRANCE_Grammar_DataLoader, self).__init__(subset, data_path, features_path, tokenizer, **kwargs)
        self.task_type = task_type
        self.grammar_mask = grammar_mask
        self.geometry = trance_grammar.geometry_of(data_path)

        # Members share the program and the members' scenes, so derivation and
        # masks are computed once per transformation.
        members = defaultdict(list)
        for sample in sorted(self.samples, key=lambda s: s["member"]):
            members[sample["trans_id"]].append(sample["objects"])
        self.groups = {}
        for sample in self.samples:
            trans_id = sample["trans_id"]
            if trans_id in self.groups:
                continue
            rules = trance_grammar.derivation_of(
                trance_grammar.program_of_transformations(sample["transformations"]))
            try:
                masks = trance_grammar.teacher_masks(grammar_mask, rules, members[trans_id], self.geometry)
            except ValueError as e:
                raise ValueError("sample {} (trans_id {}): {}".format(sample["sample_id"], trans_id, e))
            self.groups[trans_id] = {"rules": np.array(rules, dtype=np.int64), "masks": masks,
                                     "members": members[trans_id]}
        self.trans_id_of = {sample["sample_id"]: sample["trans_id"] for sample in self.samples}

    def members_of(self, sample_id):
        """Stage-0 objects of every member of the sample's transformation."""
        return self.groups[self.trans_id_of[sample_id]]["members"]

    def rules_of(self, sample_id):
        return self.groups[self.trans_id_of[sample_id]]["rules"]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        group = self.groups[sample["trans_id"]]
        rules = group["rules"]
        rule_valid = (rules != trance_grammar.NOOP).astype(np.int64)
        bef_image = self._get_rawimage(os.path.join(self.features_path, sample["bef_image"]))
        aft_image = self._get_rawimage(os.path.join(self.features_path, sample["aft_image"]))
        image_mask = np.ones(2, dtype=np.int64)
        last = np.int64(sample["target_id"]) if self.task_type == "retrieval" else sample["sample_id"]
        return rules, rule_valid, group["masks"], bef_image, aft_image, image_mask, last
