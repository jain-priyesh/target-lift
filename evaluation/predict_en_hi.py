import argparse
import dataclasses
import json
import logging
# import gc #PJ
# import time #PJ
import os #PJ
# import jsonlines #PJ
from tqdm import tqdm

from coverage.evaluator import CoverageEvaluator
from evaluation.utils import MqmDataset, MqmSample
from translation_models import load_forward_and_backward_model


def main(
    model_name: str, 
    language_pair: str, 
    split: str, 
    thresholds: str, 
    subw_agg: str, 
    detect_method: str, 
    word_agg: str, 
    temperatures: str,
    predictions_path :str,
    word_score_flag: bool
    ):
    src_lang = language_pair.split("-")[0]
    tgt_lang = language_pair.split("-")[1]

    forward_model, backward_model = load_forward_and_backward_model(model_name, src_lang, tgt_lang)

    evaluator = CoverageEvaluator(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        forward_evaluator=forward_model,
        backward_evaluator=backward_model,
        batch_size=1,
    )
    logging.warning(' Model loaded')

    data_dir = "/mnt/a99/d0/priyeshjain/dataset/English-Hindi_ErrorAnnotation"
    src_lines = open(os.path.join(data_dir, "source.en.tok"), "r").readlines()
    tgt_lines = open(os.path.join(data_dir, "translation.hi"), "r").readlines()
    '''
    for sid, (src, tgt) in tqdm(enumerate(zip(src_lines, tgt_lines))):
        if model_name == "mBart_n_mBart":
            result = evaluator.clm_detect_omm_errors( #PJ
                    src=src.strip(),
                    translation=tgt.strip(),
                    threshold=float(thresholds),
                    subw_agg=subw_agg
                )
        else:
            result = evaluator.detect_errors(
                src=src,
                translation=tgt,
            )
        
        sample = {"id" : sid}
        print(json.dumps({
            "sample": sample,
            "prediction": dataclasses.asdict(result),
        }))
        # break
    '''       
    # temperatures = list(map(float,temperatures.split()))
    thresholds = list(map(float,thresholds.split()))
    # threshold = float(thresholds)
    temperature = float(temperatures)
    # for temperature in temperatures:
    for threshold in thresholds:
        output = []
        for sid, (src, tgt) in tqdm(enumerate(zip(src_lines, tgt_lines))):
            if split == "dev":
                if sid >= 500: continue
            elif split =="test":
                if sid < 500: continue
            else:
                raise ValueError

            if model_name == "mBart_n_mBart":
                if detect_method == "relative":
                    result = evaluator.relative_clm_detect_omm_errors( #PJ
                            src=src.strip(),
                            translation=tgt.strip(),
                            threshold=threshold,
                            subw_agg=subw_agg,
                            temperature=temperature
                        )
                elif detect_method == "preceed":
                    result = evaluator.preceed_mask_clm_errors( #PJ
                            src=src.strip(),
                            translation=tgt.strip(),
                            threshold=threshold,
                            subw_agg=subw_agg,
                            temperature=temperature
                        )
                else:
                    result = evaluator.clm3_detect_omm_errors( #PJ
                            src=src.strip(),
                            translation=tgt.strip(),
                            threshold=threshold,
                            subw_agg=subw_agg,
                            temperature=temperature
                        )
                    # result =  evaluator.get_omission_error_clm_phrase(
                    #             src=src,
                    #             tgt=tgt,
                    #             threshold=threshold,
                    #             word_score=word_score_flag,
                    #             subw_agg=subw_agg,
                    #             word_agg=word_agg
                    #     )

            else:
                if detect_method == 'ratio':
                    result = evaluator.detect_omm_errors( #PJ
                        src=src,
                        translation=tgt,
                        threshold=threshold,
                        subw_agg=subw_agg
                    )

                elif detect_method == 'ww_mask':
                    result = evaluator.new_ww_mask_omission_error( #PJ
                        src=src,
                        translation=tgt,
                        threshold=threshold,
                        subw_agg=subw_agg
                    )
                elif detect_method == 'hybrid_ww_mask':
                    result = evaluator.new_ww_mask_omission_error( #PJ
                        src=src,
                        translation=tgt,
                        threshold=threshold,
                        subw_agg=subw_agg,
                        hybrid=True
                    )
                else:
                    result = evaluator.detect_errors(
                        src=src,
                        translation=tgt,
                    )
            
            sample = {"id" : sid}
            output.append(json.dumps({
                "sample": sample,
                "prediction": dataclasses.asdict(result),
            })+'\n')
            # break
        # open(os.path.join(predictions_path,f"out_{split}_thr_{threshold}.jsonl"), "w").writelines(output)
        # open(os.path.join(predictions_path,f"out_{split}_{threshold}_temp_{temperature}.jsonl"), "w").writelines(output)
        open(predictions_path, "w").writelines(output)
        # open(os.path.join(predictions_path,f"en-hi_out_all_{threshold}_temp_{temperature}.jsonl"), "w").writelines(output)
        # '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--language-pair")
    parser.add_argument("--split") #PJ
    parser.add_argument("--thresholds") #PJ
    parser.add_argument("--subword-agg") #PJ
    parser.add_argument("--detect-method") #PJ
    parser.add_argument("--word-agg") #PJ
    parser.add_argument("--temperatures") #PJ
    parser.add_argument("--word-score-flag") #PJ
    parser.add_argument("--predictions-path") #PJ
    args = parser.parse_args()
    if args.word_score_flag == "True":
        word_score_flag=True
    else:
        word_score_flag=False
    main(args.model_name, args.language_pair, args.split, args.thresholds, args.subword_agg, args.detect_method, args.word_agg, args.temperatures, args.predictions_path, word_score_flag) #PJ
