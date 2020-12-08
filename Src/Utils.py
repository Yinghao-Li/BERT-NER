import numpy as np
import torch.nn as nn
from scipy.special import softmax
from typing import List, Optional, Union, Dict, Tuple, Any
from tokenizations import get_alignments, get_original_spans
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction


def span_to_label(tokens: List[str],
                  labeled_spans: Dict[Tuple[int, int], Any]) -> List[str]:
    """
    Convert label spans to
    :param tokens: a list of tokens
    :param labeled_spans: a list of tuples (start_idx, end_idx, label)
    :return: a list of string labels
    """
    if labeled_spans:
        assert list(labeled_spans.keys())[-1][1] <= len(tokens), ValueError("label spans out of scope!")

    labels = ['O'] * len(tokens)
    for (start, end), label in labeled_spans.items():
        if type(label) == list or type(label) == tuple:
            lb = label[0][0]
        else:
            lb = label
        labels[start] = 'B-' + lb
        if end - start > 1:
            labels[start + 1: end] = ['I-' + lb] * (end - start - 1)

    return labels


def respan(src_tokens: List[str],
           tgt_tokens: List[str],
           src_span: List[tuple]):
    """
    transfer original spans to target spans
    :param src_tokens: source tokens
    :param tgt_tokens: target tokens
    :param src_span: a list of span tuples. The first element in the tuple
    should be the start index and the second should be the end index
    :return: a list of transferred span tuples.
    """
    s2t, _ = get_alignments(src_tokens, tgt_tokens)
    tgt_spans = list()
    for spans in src_span:
        start = s2t[spans[0]][0]
        if spans[1] < len(s2t):
            end = s2t[spans[1]][-1]
        else:
            end = s2t[-1][-1]
        if end == start:
            end += 1
        tgt_spans.append((start, end))

    return tgt_spans


def txt_to_token_span(tokens: List[str],
                      text: str,
                      txt_spans):
    """
    Transfer text-domain spans to token-domain spans
    :param tokens: tokens
    :param text: text
    :param txt_spans: text spans tuples: (start, end, ...)
    :return: a list of transferred span tuples.
    """
    token_indices = get_original_spans(tokens, text)
    tgt_spans = list()
    for txt_span in txt_spans:
        txt_start = txt_span[0]
        txt_end = txt_span[1]
        start = None
        end = None
        for i, (s, e) in enumerate(token_indices):
            if s <= txt_start < e:
                start = i
            if s <= txt_end <= e:
                end = i + 1
            if (start is not None) and (end is not None):
                break
        assert (start is not None) and (end is not None), ValueError("input spans out of scope")
        tgt_spans.append((start, end))
    return tgt_spans


def token_to_txt_span(tokens: List[str],
                      text: str,
                      token_spans):
    """
    Transfer text-domain spans to token-domain spans
    :param tokens: tokens
    :param text: text
    :param token_spans: text spans tuples: (start, end, ...)
    :return: a list of transferred span tuples.
    """
    token_indices = get_original_spans(tokens, text)
    tgt_spans = dict()
    for token_span, value in token_spans.items():
        txt_start = token_indices[token_span[0]][0]
        txt_end = token_indices[token_span[1]-1][1]
        tgt_spans[(txt_start, txt_end)] = value
    return tgt_spans


def label_to_span(labels: List[str],
                  scheme: Optional[str] = 'BIO') -> dict:
    """
    convert labels to spans
    :param labels: a list of labels
    :param scheme: labeling scheme, in ['BIO', 'BILOU'].
    :return: labeled spans, a list of tuples (start_idx, end_idx, label)
    """
    assert scheme in ['BIO', 'BILOU'], ValueError("unknown labeling scheme")

    labeled_spans = dict()
    i = 0
    while i < len(labels):
        if labels[i] == 'O':
            i += 1
            continue
        else:
            if scheme == 'BIO':
                if labels[i][0] == 'B':
                    start = i
                    lb = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] == 'I':
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = lb
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = lb
                        i += 1
                # this should not happen
                elif labels[i][0] == 'I':
                    i += 1
            elif scheme == 'BILOU':
                if labels[i][0] == 'U':
                    start = i
                    end = i + 1
                    lb = labels[i][2:]
                    labeled_spans[(start, end)] = lb
                    i += 1
                elif labels[i][0] == 'B':
                    start = i
                    lb = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] != 'L':
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = lb
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = lb
                        break
                    i += 1
                else:
                    i += 1

    return labeled_spans


def align_predictions(predictions: np.ndarray,
                      label_ids: np.ndarray,
                      label_map: Dict):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(p: EvalPrediction, label_map: Dict[int, str]) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map=label_map)
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def soft_frequency(logits, power=2, probs=False):
    """
    Unsupervised Deep Embedding for Clustering Analysis
    https://arxiv.org/abs/1511.06335
    """
    if not probs:
        p = softmax(logits, axis=-1)
        p[p != p] = 0
    else:
        p = logits
    f = np.sum(p, axis=0, keepdims=True)
    p = p ** power / f
    p = p / np.sum(p, axis=-1, keepdims=True) + 1e-9
    p[p != p] = 0

    return p