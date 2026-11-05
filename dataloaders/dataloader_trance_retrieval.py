from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
from dataloaders.rawimage_util import RawImageExtractor
from dataloaders.trance_util import load_trance_split


class TRANCE_DataLoader(Dataset):
    """TRANCE dataset loader (retrieval). The text is a rendering of the JSON transformation.

    Items have a 7th element after the Spot layout: the target_id of the sample.
    """
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            image_resolution=224,
            final_camera="Camera_Center",
            test_transformation_generalization=False,
            key="text",
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.max_words = max_words
        self.tokenizer = tokenizer

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        # key: what makes two samples the same transformation (trance_util.KEYS); it must follow the
        # target, so the grammar loader passes "program"
        samples = load_trance_split(self.data_path, self.subset, final_camera=final_camera,
                                    test_transformation_generalization=test_transformation_generalization,
                                    key=key)
        self.samples = samples
        # members of one transformation share a trans_id (every sample is its own group when there is one member)
        self.trans_ids = [sample["trans_id"] for sample in samples]
        # samples with the same target_id have the same target (by key), also across groups; the
        # contrastive loss does not use them as negatives of each other, and eval scores groups against
        # distinct targets
        self.target_ids = [sample["target_id"] for sample in samples]

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []

        for sample in samples:
            self.sentences_dict[len(self.sentences_dict)] = (sample["sample_id"], sample["text"])
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.image_num: used to cut the image pair representation
        self.multi_sentence_per_pair = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.image_num = len(samples)
            assert len(self.cut_off_points) == self.image_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, image number: {}".format(self.subset, self.image_num))

        print("Image number: {}".format(len(samples)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawImageExtractor = RawImageExtractor(size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, image_id, caption):
        k = 1
        choice_image_ids = [image_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, image_id in enumerate(choice_image_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_rawimage(self, image_path):
        choice_image_path = [image_path]
        # Pair x L x T x 3 x H x W
        image = np.zeros((len(choice_image_path), 3, self.rawImageExtractor.size,
                          self.rawImageExtractor.size), dtype=float)

        for i, image_path in enumerate(choice_image_path):

            raw_image_data = self.rawImageExtractor.get_image_data(image_path)
            raw_image_data = raw_image_data['image']

            image[i] = raw_image_data

        return image

    def __getitem__(self, idx):
        image_id, caption = self.sentences_dict[idx]
        sample = self.samples[idx]
        bef_image_path = os.path.join(self.features_path, sample["bef_image"])
        aft_image_path = os.path.join(self.features_path, sample["aft_image"])

        pairs_text, pairs_mask, pairs_segment = self._get_text(image_id, caption)
        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        image_mask = np.ones(2, dtype=np.int64)
        target_id = np.int64(sample["target_id"])
        return pairs_text, pairs_mask, pairs_segment, bef_image, aft_image, image_mask, target_id
