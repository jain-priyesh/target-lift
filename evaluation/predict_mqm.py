import argparse
import dataclasses
import json

from tqdm import tqdm

from coverage.evaluator import CoverageEvaluator
from evaluation.utils import MqmDataset, MqmSample
from translation_models import load_forward_and_backward_model


def main(model_name: str, 
        language_pair: str,
        subword_agg: str, 
        detect_method: str, 
        temperature: float,
        ):
    src_lang = language_pair.split("-")[0]
    tgt_lang = language_pair.split("-")[1]

    dataset = MqmDataset(language_pair)

    causallm_model, reverse_nmt_model = load_forward_and_backward_model(model_name, src_lang, tgt_lang)

    evaluator = CoverageEvaluator(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        causallm_evaluator=causallm_model,
        reverse_nmt_evaluator=reverse_nmt_model,
        batch_size=1,
    )

    for sample in tqdm(dataset.load_samples(load_original_sequences=True)):
        sample: MqmSample = sample

        if detect_method == "relative":
            result = evaluator.generate_scores( 
                        src=sample.original_source,
                        translation=sample.original_target,
                        subword_agg=subword_agg,
                        temperature=temperature
                    )
        elif detect_method == "preceed":
            result = evaluator.generate_scores( 
                        src=sample.original_source,
                        translation=sample.original_target,
                        subword_agg=subword_agg,
                        temperature=temperature
                    )
        else:
            result = evaluator.generate_scores( 
                        src=sample.original_source,
                        translation=sample.original_target,
                        subword_agg=subword_agg,
                        temperature=temperature
                    )

        print(json.dumps({
            "sample": dataclasses.asdict(sample),
            "prediction": dataclasses.asdict(result),
        }))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--language-pair")
    parser.add_argument("--subword-agg") 
    parser.add_argument("--detect-method") 
    parser.add_argument("--temperature") 
    args = parser.parse_args()

    if args.word_score_flag == "True":
        word_score_flag=True
    else:
        word_score_flag=False

    main(args.model_name, 
        args.language_pair, 
        args.subword_agg, 
        args.detect_method, 
        float(args.temperature))