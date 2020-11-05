import logging
import os
import torch
import json
from typing import List, TextIO, Union, Optional

from Src.Data import InputExample, Split, TokenClassificationTask
from Src.Util import span_to_label

logger = logging.getLogger(__name__)


class NER(TokenClassificationTask):
    def __init__(self, label_idx=-1):
        # in NER datasets, the last column is usually reserved for NER label
        self.label_idx = label_idx

    def read_examples_from_file(
            self,
            data_dir: str,
            dataset: str,
            mode: Union[Split, str],
            weak_src: Optional[str] = None) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{dataset}-linked-{mode}.pt")
        weak_lbs_file_path = os.path.join(data_dir, f"{dataset}-linked-{mode}.{weak_src}.scores") \
            if weak_src is not None else None
        data = torch.load(file_path)
        words_list = data['sentences']
        spanned_labels_list = data['labels']
        weak_lbs_list = torch.load(weak_lbs_file_path)[1] if weak_src else \
            [None for _ in range(len(words_list))]
        examples = []
        for guid_index, (words, spanned_lbs, weak_lbs) in \
                enumerate(zip(words_list, spanned_labels_list, weak_lbs_list)):
            if weak_lbs is not None:
                assert len(words) == len(weak_lbs)
            lbs = span_to_label(words, spanned_lbs)
            assert len(words) == len(lbs)
            examples.append(InputExample(
                guid=f"{mode}-{guid_index+1}", words=words, labels=lbs, weak_lb_weights=weak_lbs
            ))
        return examples

    @staticmethod
    def write_predictions_to_file(writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)
            else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def get_labels(self, args) -> List[str]:
        with open(os.path.join(args.data_dir, args.dataset_name, f"{args.dataset_name}-metadata.json")) as f:
            lbs = json.load(f)['labels']
        bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in lbs for bi in "BI"]
        return bio_lbs

