"""Write the COCO-format ground truth that main_task_caption.py scores against (--gt_dir <out_dir>).

The references are produced by the same function as the training targets
(dataloaders.trance_util.transformation_to_text), so they stay in sync when that function is replaced.

Usage: python preprocess/trance_gt.py <data_dir> [out_dir]    (out_dir defaults to data_dir)
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.trance_util import load_trance_split

if __name__ == "__main__":
    data_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else data_dir
    with open(os.path.join(data_dir, "split.json"), "r") as fp:
        split_names = json.load(fp)["split_name"]

    images, annotations = [], []
    for name in split_names:
        for sample in load_trance_split(data_dir, name):
            images.append({"filename": sample["aft_image"], "id": sample["sample_id"]})
            annotations.append({"caption": sample["text"], "id": len(annotations), "image_id": sample["sample_id"]})

    out_file = os.path.join(out_dir, "trance_total_change_captions_reformat.json")
    with open(out_file, "w") as fp:
        json.dump({"info": {}, "licenses": [], "type": "captions", "images": images, "annotations": annotations}, fp)
    print("Wrote {} images / {} annotations to {}".format(len(images), len(annotations), out_file))
