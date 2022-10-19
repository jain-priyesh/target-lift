#!/bin/bash

gpu_no=$1
export CUDA_VISIBLE_DEVICES="${gpu_no}"

model_name="mBart_n_mBart"   
lang_pair="en-de"      
subword_agg="first"               
detect_method="difference"
temperature=1.0

split="test"
threshold=0.11    
predictions_path="predictions/out_${lang_pair}_${subword_agg}_${temperature}.jsonl"  
results_path="predictions/scores_${lang_pair}_${subword_agg}_${temperature}_${split}_${threshold}.jsonl" 


echo "Predicting  for subword_agg=${subword_agg} temperature=${temperature} ..."
python -m evaluation.predict_mqm \
  --model-name "${model_name}" \
  --language-pair "${lang_pair}" \
  --subword-agg "${subword_agg}" \
  --detect-method "${detect_method}" \
  --temperature "${temperature}" \
  > "${predictions_path}" 

echo "Evaluating (P-R-F) for split=${split} subword_agg=${subword_agg} thr=${threshold} ..."
python -m evaluation.evaluate_mqm \
  --language-pair "${lang_pair}" \
  --split "${split}" \
  --threshold "${threshols}" \
  --predictions-path "${predictions_path}" \
  > "${results_path}" 

