import logging
import os
import torch
import json
import numpy as np
from typing import List, Union, Optional
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from Src.Data import InputExample, Split, TokenClassificationTask
from Src.Util import span_to_label
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class NER(TokenClassificationTask):
    def __init__(self):
        # in NER datasets, the last column is usually reserved for NER label
        self._mappings = None
        self._lbs = None

    def read_examples_from_file(
            self,
            data_dir: str,
            dataset: str,
            tokenizer: PreTrainedTokenizer,
            mode: Union[Split, str],
            max_seq_length: Optional[int] = 512,
            weak_src: Optional[str] = None) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{dataset}-linked-{mode}.pt")
        weak_lbs_file_path = os.path.join(data_dir, f"{dataset}-linked-{mode}.{weak_src}.scores") \
            if weak_src else None
        data = torch.load(file_path)
        words_list = data['sentences']
        spanned_labels_list = data['labels']
        if dataset == 'Co03':
            weak_lbs_list, weak_scores = torch.load(weak_lbs_file_path)
            if weak_scores[0].shape[1] == 39:
                temp_spans = list()
                for spans in weak_lbs_list:
                    normalized_span = dict()
                    for k, v in spans.items():
                        norm_span = self._mappings.get(v[0][0], v[0][0])
                        if norm_span in self._lbs:
                            normalized_span[k] = norm_span
                    temp_spans.append(normalized_span)
                weak_lbs_list = temp_spans
        else:
            weak_lbs_list = torch.load(weak_lbs_file_path)[1] if weak_src else \
                [None for _ in range(len(words_list))]
        examples = []

        # TODO: Divide the sequence that exceeds the maximum length into several instances
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        buffer_length = 20
        max_token_len = max_seq_length - special_tokens_count - buffer_length

        tmp_word_list = list()
        tmp_span_list = list()
        tmp_score_list = list()
        for words, spanned_lbs, weak_lbs in zip(words_list, spanned_labels_list, weak_lbs_list):

            nltk_string = ' '.join(words)
            len_bert_tokens = len(tokenizer.tokenize(nltk_string))

            if len_bert_tokens >= max_token_len:
                if dataset == 'Co03':
                    raise NotImplementedError(
                        'This function has not been implemented for CoNLL 2003 dataset. '
                        'Please use a larger maximum sequence length!'
                    )
                nltk_tokens_list = [words]
                span_list = [spanned_lbs]
                score_list = [weak_lbs] if weak_lbs is not None else [None for _ in range(len(words))]
                bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in nltk_tokens_list]

                while (np.asarray(bert_length_list) >= max_token_len).any():
                    new_token_list = list()
                    new_span_list = list()
                    new_score_list = list()
                    for nltk_tokens, spans, scores, bert_len in \
                            zip(nltk_tokens_list, span_list, score_list, bert_length_list):
                        if bert_len < max_token_len:
                            new_token_list.append(nltk_tokens)
                            span_list.append(spans)
                            score_list.append(scores)
                            continue

                        # split sentences
                        nltk_string = ' '.join(nltk_tokens)
                        sts = sent_tokenize(nltk_string)

                        sent_lens = list()
                        for st in sts:
                            sent_lens.append(len(word_tokenize(st)))
                        ends = [np.sum(sent_lens[:i]) for i in range(1, len(sent_lens) + 1)]

                        nearest_end_idx = int(np.argmin((np.array(ends) - len(nltk_tokens) / 2) ** 2))

                        split_idx = ends[nearest_end_idx]
                        token_split_1 = nltk_tokens[:split_idx]
                        token_split_2 = nltk_tokens[split_idx:]
                        new_token_list.extend([token_split_1, token_split_2])
                        if scores is not None:
                            score_split_1 = scores[:split_idx]
                            score_split_2 = scores[split_idx:]
                            new_score_list.extend([score_split_1, score_split_2])

                        # deal with spans according to splitted sentences
                        ns_1 = dict()
                        ns_2 = dict()
                        for span, v in spans.items():
                            if span[0] < split_idx and span[1] <= split_idx:
                                ns_1[span] = v
                            elif span[0] >= split_idx and span[1] <= len(nltk_tokens):
                                ns_2[(span[0]-split_idx, span[1]-split_idx)] = v

                        new_span_list.extend([ns_1, ns_2])

                    nltk_tokens_list = new_token_list
                    span_list = new_span_list
                    score_list = new_score_list
                    bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in nltk_tokens_list]
                tmp_word_list.extend(nltk_tokens_list)
                tmp_span_list.extend(span_list)
                tmp_score_list.extend(score_list)
            else:
                tmp_word_list.append(words)
                tmp_span_list.append(spanned_lbs)
                tmp_score_list.append(weak_lbs)
        words_list = tmp_word_list
        spanned_labels_list = tmp_span_list
        if weak_src:
            weak_lbs_list = tmp_score_list

        for guid_index, (words, spanned_lbs, weak_lbs) in \
                enumerate(zip(words_list, spanned_labels_list, weak_lbs_list)):
            if weak_lbs is not None and dataset != 'Co03':
                assert len(words) == len(weak_lbs)
            lbs = span_to_label(words, spanned_lbs)
            assert len(words) == len(lbs)
            if dataset == 'Co03':
                weak_lbs = span_to_label(words, weak_lbs)
                assert len(words) == len(weak_lbs)
            examples.append(InputExample(
                guid=f"{mode}-{guid_index+1}", words=words, labels=lbs, weak_lb_weights=weak_lbs
            ))
        return examples

    def get_labels(self, args) -> List[str]:
        with open(os.path.join(args.data_dir, args.dataset_name, f"{args.dataset_name}-metadata.json")) as f:
            metadata = json.load(f)
        lbs = metadata['labels']
        if 'mapping' in metadata.keys():
            self._mappings = metadata['mapping']
            self._lbs = lbs
        bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in lbs for bi in "BI"]
        return bio_lbs
