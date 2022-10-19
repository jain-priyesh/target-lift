import argparse
from pathlib import Path
from ssl import OPENSSL_VERSION_INFO
from typing import Union
from unittest import skip

import jsonlines
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, confusion_matrix #PJ
import json #PJ
from time import sleep #PJ
import os
from tqdm import tqdm

from coverage.evaluator import CoverageResult
from evaluation.utils import MqmDataset, MqmSample
from metric import precision, recall, f1, tpr, fpr, true_negative, true_positive, false_negative, false_positive #PJ


def main(language_pair: str, split: str, thresholds :str, temperatures :str, predictions_path: Union[Path, str], cv: str = None):

    
    data_dir = "/mnt/a99/d0/priyeshjain/dataset/English-Hindi_ErrorAnnotation"
    src_lines = open(os.path.join(data_dir, "source.en.tok"), "r").readlines()
    flag_lines = open(os.path.join(data_dir, "new_flags.txt"), "r").readlines()
    # skip_ids = list(map(int,open("predictions/en-hi/skip_dev_ids_en_hi.txt","r").readlines())) #PJ
    '''
    true_omissions = []
    predicted_omissions = []
    # output = []
    unequal_count = 0
    multi_sentence_count = 0
    counter = 0
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    overall_tpr = []
    overall_fpr = []
    with jsonlines.open(predictions_path) as f:
        for line in tqdm(f):
            sample_id = line["sample"]["id"]
            # if sample_id in skip_ids: continue
            # if sample_id in [0,1]: continue
            if split == "dev":
                if sample_id >= 500: continue
            elif split =="test":
                if sample_id < 500: continue
            else:
                raise ValueError
            prediction = CoverageResult.from_dict(line["prediction"])
            # Exclude multi-sentence samples
            if prediction.is_multi_sentence_input:
                multi_sentence_count += 1
                continue
            src = src_lines[sample_id]
            # print(src)
            src_toks = [src.split()[0]] + [' '+tok for tok in src.split()[1:]]
            # print(src_toks)
            preds = [0 for _ in range(len(src_toks))]
            
            # if len(line["prediction"]["omission_errors"]) >= 1:
            #     # for index in line["prediction"]["omission_errors"]:
            #     #     preds[index] = 1
            #     for omission in line["prediction"]["omission_errors"]:
            #         start_char = int(omission["constituent"]["start_char"])
            #         end_char = int(omission["constituent"]["end_char"])
            #         pointer = 0
            #         start_found = False
            #         end_found = False
            #         idx = 0
            #         start_idx, end_idx = None, None
            #         while idx < len(src_toks) and ((not start_found) or (not end_found)):
            #             pointer += len(src_toks[idx])
            #             # print(f"idx: {idx:3}\tpointer: {pointer:3}")
            #             if not start_found:
            #                 if pointer >= start_char:
            #                     start_idx = idx
            #                     start_found = True
            #             if start_found and not end_found:
            #                 if pointer >= end_char:
            #                     end_idx = idx
            #                     end_found = True
            #             idx += 1
                
            #         if start_idx != None and end_idx != None:
            #             for val in range(start_idx, end_idx+1):
            #                 preds[val] = 1
            #             # print(sample_id,''.join(src_toks[start_idx: end_idx+1]))
            #             # print(sample_id, [1-x for x in list(map(int,flag_lines[sample_id].strip().split()))])
            #     # sleep(20)
                           
            true_labels = list(map(int,flag_lines[sample_id].strip().split()))
            # preds = [1-x for x in prediction.omission_errors]
            for val in prediction.omission_errors:
                preds[val] = 1
            if len(true_labels) != len(preds):
                unequal_count += 1
                continue
            
            predicted_omissions.extend(preds)
            # predicted_omissions.append(0 in prediction.omission_errors)
            true = [1-x for x in true_labels]
            true_omissions.extend(true)
            # print(sample_id, preds)
            # print(sample_id, true,'\n')
            # print(sample_id,'\n',true,'\n',preds)
            # true_omissions.append(0 in true_labels)
            # result = precision_recall_fscore_support(np.array(true), np.array(true), average='binary')
            r_precision = precision(np.array(true).astype(bool), np.array(preds).astype(bool))
            r_recall = recall(np.array(true).astype(bool), np.array(preds).astype(bool))
            r_f1= f1(np.array(true).astype(bool), np.array(preds).astype(bool))
            r_tpr = tpr(np.array(true).astype(bool), np.array(preds).astype(bool))
            r_fpr = fpr(np.array(true).astype(bool), np.array(preds).astype(bool))
            overall_precision.append(r_precision)
            overall_recall.append(r_recall)
            overall_f1.append(r_f1)
            overall_tpr.append(r_tpr)
            overall_fpr.append(r_fpr)
            # print(f"Precision\tRecall\tF1\tTPR\tFPR")
            # print(f"    {result[0]:1.3f}\t {result[1]:1.3f}\t{result[2]:.3f}\t{r_tpr:.3f}\t{r_fpr:.3f}")
            # print(f"    {r_precision:1.3f}\t {r_recall:1.3f}\t{r_f1:.3f}\t{r_tpr:.3f}\t{r_fpr:.3f}")
            # if sample_id > 10: break
    print(f"Precision: {sum(overall_precision)/len(overall_precision)*100:1.3f}",end="\t")
    print(f"Recall: {sum(overall_recall)/len(overall_recall)*100:1.3f}",end="\t")
    print(f"f1: {sum(overall_f1)/len(overall_f1)*100:1.3f}")
    print(f"|============================================|")
    print(f"| Omission errors                            |")
    assert len(true_omissions) == len(predicted_omissions)
    print(f"|____________________________________________|")
    print(f"| Number of instances: {len(overall_f1):4}                  |")
    print(f"|____________________________________________|")
    result = precision_recall_fscore_support(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool), average='binary')
    r_tpr = tpr(np.array(true_omissions), np.array(predicted_omissions))
    r_fpr = fpr(np.array(true_omissions), np.array(predicted_omissions))
    print(f"Precision\tRecall\tF1\tTPR\tFPR")
    print(f"    {result[0]*100:2.3f}\t {result[1]*100:2.3f}\t{result[2]*100:2.3f}\t{r_tpr*100:2.3f}\t{r_fpr*100:2.3f}")
    print(f"Precision: {precision(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool)):1.3f}")
    print(f"Recall: {recall(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool)):1.3f}")
    print(f"f1: {f1(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool)):1.3f}")
    print(f"unequal_count: {unequal_count}") 
    print(f"multi_sentence_count : {multi_sentence_count }")
    print(confusion_matrix(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool)))
    print(true_omissions.count(1),"/",len(true_omissions))
    print(predicted_omissions.count(1),"/",len(predicted_omissions))
    tp=true_positive(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool))
    tn=true_negative(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool))
    fp=false_positive(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool))
    fn=false_negative(np.array(true_omissions).astype(bool), np.array(predicted_omissions).astype(bool))
    print(f"\nConfusion matrix: ")
    print(f"\tTP:{tp:5}\tFN:{fn:5}")
    print(f"\tFP:{fp:5}\tTN:{tn:5}\n")
    # print(true_omissions)
    # print(predicted_omissions)
    
    # open(os.path.join(predictions_path,f"PRF2_{split}_thr_{threshold}.txt"), "w").writelines(output)
    '''
    
    # temperatures = list(map(float,temperatures.split()))
    thresholds = list(map(float,thresholds.split()))
    # threshold = float(thresholds)
    temperature = float(temperatures)
    # for temperature in temperatures:
    for threshold in thresholds:
        true_omissions = []
        predicted_omissions = []
        output = []
        unequal_count = 0
        multi_sentence_count = 0
        counter = 0
        # predictions_file = os.path.join(predictions_path,f"out_{split}_{threshold}_temp_{temperature}.jsonl")
        predictions_file = predictions_path
        with jsonlines.open(predictions_file) as f:
            for line in tqdm(f):
                sample_id = line["sample"]["id"]
                if split == "dev":
                    if sample_id >= 500: continue
                elif split =="test":
                    if sample_id < 500: continue
                else:
                    raise ValueError
                prediction = CoverageResult.from_dict(line["prediction"])
                # Exclude multi-sentence samples
                if prediction.is_multi_sentence_input:
                    multi_sentence_count += 1
                    continue
                # src = src_lines[sample_id]
                # # print(src)
                # src_toks = [src.split()[0]] + [' '+tok for tok in src.split()[1:]]
                # print(src_toks)
                # preds = [0 for _ in range(len(src_toks))]
                # if len(line["prediction"]["omission_errors"]) >= 1:
                #     for omission in line["prediction"]["omission_errors"]:
                #         start_char = int(omission["constituent"]["start_char"])
                #         end_char = int(omission["constituent"]["end_char"])
                #         pointer = 0
                #         start_found = False
                #         end_found = False
                #         idx = 0
                #         start_idx, end_idx = None, None
                #         while idx < len(src_toks) and ((not start_found) or (not end_found)):
                #             pointer += len(src_toks[idx])
                #             # print(f"idx: {idx:3}\tpointer: {pointer:3}")
                #             if not start_found:
                #                 if pointer >= start_char:
                #                     start_idx = idx
                #                     start_found = True
                #             if start_found and not end_found:
                #                 if pointer >= end_char:
                #                     end_idx = idx
                #                     end_found = True
                #             idx += 1
                    
                #         if start_idx != None and end_idx != None:
                #             for val in range(start_idx, end_idx+1):
                #                 # preds[val] = 1
                #                 p_score = omission["constituent"]["constituent_score"] - omission["constituent"]["base_score"]
                #                 if p_score > preds[val]:
                #                     preds[val] = p_score
                #             # print(''.join(src_toks[start_idx: end_idx+1]))
                #             # print(sample_id, preds)
                #             # print(sample_id, [1-x for x in list(map(int,flag_lines[sample_id].strip().split()))])
                    # sleep(30)
                # if cv: 
                #     counter += 1 #PJ
                #     if cv == 'cv1':
                #         if counter < 236: continue
                #     if cv == 'cv2':
                #         if counter > 235 and counter < 471: continue
                #     if cv == 'cv3':
                #         if counter > 470: continue
                # src = src_lines[sample_id]    
                # src_toks = [src.split()[0]] + [' '+tok for tok in src.split()[1:]]
                # print(src_toks)
                # preds = [0 for _ in range(len(src_toks))] #PJ
                preds = prediction.omission_scores
                true_labels = list(map(int,flag_lines[sample_id].strip().split()))
                # preds = [0 for _ in range(len(src.strip().split()))]
                # preds = [1-x for x in prediction.omission_errors]
                # for val in prediction.omission_errors: #PJ
                #     preds[val] = 1 #PJ
                if len(true_labels) != len(preds):
                    unequal_count += 1
                    continue
                # print(preds)
                # print(preds)
                # sleep(100)
                predicted_omissions.extend(preds)
                # predicted_omissions.append(0 in prediction.omission_errors)
                true = [1-x for x in true_labels]
                true_omissions.extend(true)
                # print(true,'\n',preds)

                # true_omissions.append(0 in true_labels)
        output.append(f"|============================================|\n")
        output.append(f"| Omission errors                            |\n")
        assert len(true_omissions) == len(predicted_omissions)
        output.append(f"|____________________________________________|\n")
        output.append(f"| Number of instances: {len(predicted_omissions):4}                  |\n")
        output.append(f"|____________________________________________|\n")
        # result = precision_recall_fscore_support(np.array(true_omissions), np.array(predicted_omissions), average='binary')
        # r_tpr = tpr(np.array(true_omissions), np.array(predicted_omissions))
        # r_fpr = fpr(np.array(true_omissions), np.array(predicted_omissions))
        # output.append(f"Precision\tRecall\tF1{' ':4}\tTPR{' ':4}\tFPR (out of 100)\n")
        # output.append(f"{result[0]*100:2.3f}   \t{result[1]*100:2.3f}\t{result[2]*100:2.3f}\t{r_tpr*100:.3f}\t{r_fpr*100:.3f}\n")
        output.append(f"unequal_count: {unequal_count}\n") 
        # output.append(f"multi_sentence_count : {multi_sentence_count }\n")
        # print(true_omissions)
        # print(predicted_omissions)
        for t,p in zip(true_omissions,predicted_omissions):
            output.append(f"{int(t)},{p}\n")
        
        # open(os.path.join(predictions_path,f"PRF_{split}_{threshold}_temp_{temperature}.txt"), "w").writelines(output)
        #PJ
        # open(predictions_path.split('.')[0].split('_')[1:]+'.txt',f"PRF_{split}_{threshold}_temp_{temperature}.txt"), "w").writelines(output)
        results_path = predictions_file.replace('out','scores')
        open(results_path, "w").writelines(output)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-pair")
    parser.add_argument("--split")
    parser.add_argument("--thresholds")
    parser.add_argument("--temperatures")
    parser.add_argument("--predictions-path")
    parser.add_argument("--cv") #PJ
    args = parser.parse_args()
    main(args.language_pair, args.split, args.thresholds, args.temperatures, args.predictions_path, args.cv)
