import argparse
from pathlib import Path
from time import sleep
from typing import Union

import jsonlines
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from coverage.evaluator import CoverageResult
from evaluation.utils import MqmDataset, MqmSample, get_start_end_index
from metric import tpr, fpr ,precision, recall, f1, true_negative, true_positive, false_negative, false_positive#PJ
import json #PJ
import torch

def main(language_pair: str, split: str, predictions_path: Union[Path, str], threshold: float = None):
    predictions_path = Path(predictions_path)
    assert predictions_path.exists()

    if language_pair == "en-de":
        DEV_SYSTEM = "Online-B.1590"
    elif language_pair == "zh-en":
        DEV_SYSTEM = "Online-B.1605"
    else:
        raise ValueError

    dataset = MqmDataset(language_pair)
    annotations = dataset.load_annotations()

    annotated_omissions = []
    predicted_errors = []

    nontranslations_count = 0
    five_count = 0
    quotes_count = 0
    overlong_count = 0
    multi_sentence_count = 0
    both_count = 0
    counter = 0 #PJ
    neg_samples = 0
    
    with jsonlines.open(predictions_path) as f:
        for line in tqdm(f):
            sample = MqmSample.from_dict(line["sample"])
            prediction = CoverageResult.from_dict(line["prediction"])
            annotated_sample = annotations[sample.id]
            if split == "dev":
                if sample.id.system != DEV_SYSTEM:
                    continue
            elif split == "test":
                if sample.id.system == DEV_SYSTEM:
                    continue
            else:
                raise ValueError

            # if split == 'dev':
            #     if line['sample']['id']['seg_id'] in skip_ids: #PJ
            #         continue
            # Exclude human "systems"
            if "human" in sample.id.system.lower():
                continue

            # Exclude samples with incomplete annotations
            if annotated_sample.has_any_nontranslation_rating:
                nontranslations_count += 1
                continue
            if annotated_sample.might_have_unmarked_coverage_error:
                five_count += 1
                continue

            # Exclude samples with a presumed data processing error regarding quotes
            if sample.has_superfluous_quotes:
                quotes_count += 1
                continue

            # Exclude multi-sentence samples
            if prediction.is_multi_sentence_input:
                multi_sentence_count += 1
                continue

            # Exclude very long samples
            if len(sample.original_source.split()) > 150 or len(sample.original_target.split()) > 150:
                overlong_count += 1
                continue

            if annotated_sample.has_addition_error_by_any_rater and annotated_sample.has_omission_error_by_any_rater:
                both_count += 1
                continue
            counter += 1
            annotated_omissions.append(annotated_sample.has_omission_error_by_any_rater) #PJ
            scores = np.array(prediction.src_token_scores)
            # predicted_errors.append(scores[scores > threshold].tolist())
            predicted_errors.append([min(scores)])
            if annotated_sample.has_omission_error_by_any_rater:
                neg_samples += 1
    print(f"|============================================|")
    print("| Omission errors                            |")
    assert len(annotated_omissions) == len(predicted_errors)
    print(f"|____________________________________________|")
    print(f"| Number of sentences: {counter:4}                  |")
    print(f"|____________________________________________|")
    result = precision_recall_fscore_support(np.array(annotated_omissions).astype(bool), np.array(predicted_omissions).astype(bool), average='binary')
    r_tpr = tpr(np.array(annotated_omissions).astype(bool), np.array(predicted_omissions).astype(bool))
    r_fpr = fpr(np.array(annotated_omissions).astype(bool), np.array(predicted_omissions).astype(bool))
    print(f"Precision\tRecall\tF1{' ':4}\tTPR{' ':4}\tFPR (out of 100)")
    print(f"{result[0]*100:2.3f}   \t{result[1]*100:2.3f}\t{result[2]*100:2.3f}\t{r_tpr*100:.3f}\t{r_fpr*100:.3f}")
    print(f"nontranslations_count: {nontranslations_count }") 
    print(f"five_count           : {five_count }")
    print(f"quotes_count         : {quotes_count }") 
    print(f"overlong_count       : {overlong_count }")
    print(f"multi_sentence_count : {multi_sentence_count }")
    print(f"both_count           : {both_count }")
    print(f"pos_samples          : {neg_samples}/{counter}")
    for t,p in zip(annotated_omissions,predicted_errors):
        print(f"{int(t)},{p[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-pair")
    parser.add_argument("--split")
    parser.add_argument("--predictions-path")
    parser.add_argument("--threshold") #PJ
    args = parser.parse_args()
    main(args.language_pair, args.split, args.predictions_path, args.threshold)
