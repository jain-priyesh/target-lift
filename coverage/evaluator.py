import logging
from queue import Empty
# from optparse import Option
import torch #PJ
# from nltk.corpus import stopwords
# from sre_parse import Tokenizer
from dataclasses import dataclass
from typing import List, Optional

from stanza.models.common.doc import Document

from coverage.parser import Constituent, ParseTree
from coverage.utils import load_stanza_parser
from translation_models import TranslationModel
# from transformers.file_utils import PaddingStrategy #PJ
from evaluation.utils import get_bpe_map, get_word_prob, T_scaling #PJ
from transformers.models.mbart.modeling_mbart import shift_tokens_right #PJ

try:
    from dict_to_dataclass import DataclassFromDict, field_from_dict
except:  # Python 3.7 compatibility
    DataclassFromDict = object
    def field_from_dict(default=None):
        return default

EPSILON = 1e-07 #PJ
@dataclass
class CoverageError(DataclassFromDict):
    constituent: Constituent = field_from_dict()

    def __str__(self):
        return self.constituent.removed


@dataclass
class AdditionError(CoverageError):
    pass


@dataclass
class OmissionError(CoverageError):
    pass


@dataclass
class CoverageResult(DataclassFromDict):
    src_lang: Optional[str] = field_from_dict()
    tgt_lang: Optional[str] = field_from_dict()
    subword_agg: Optional[str] = field_from_dict(default='first')
    temperature: Optional[float] = field_from_dict(default=1.0)
    is_multi_sentence_input: Optional[bool] = field_from_dict(default=None)
    src_token_scores: Optional[List[float]] = field_from_dict()
    src_tokens: Optional[List[str]] = field_from_dict()

    @property
    def contains_error(self, ) -> bool:
        return len(self.omission_errors) >= 1

    def __str__(self):
        return "".join([
            f"Source tokens: {' | '.join(map(str, self.src_tokens))}" if self.src_tokens else "",
            "\n",
            f"Token scores: {' | '.join(map(str, self.src_token_scores))}" if self.src_token_scores else "",
        ])


class CoverageEvaluator:

    def __init__(self,
                 src_lang: str = None,
                 tgt_lang: str = None,
                 forward_evaluator: TranslationModel = None,
                 backward_evaluator: TranslationModel = None,
                 batch_size: int = 16,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_parser = load_stanza_parser(src_lang) if src_lang is not None else None #PJ
        self.forward_evaluator = forward_evaluator
        self.backward_evaluator = backward_evaluator
        self.batch_size = batch_size
        # self.stopwords_toks = [''.join(self.forward_evaluator.tokenizer.tokenize(word)) for word in stopwords.words('english')]
    
    def generate_scores(self, src: str, translation: str, 
                                src_doc: Document = None, translation_doc: Document = None, 
                                subword_agg: str = 'first',
                                temperature: float = 1.0) -> CoverageResult:
        is_multi_sentence_input = False
        if src_doc is None:
            src_doc = self.src_parser(src)
        if len(src_doc.sentences) > 1:
            # logging.warning("Coverage detection currently does not handle multi-sentence inputs; skipping ...")
            is_multi_sentence_input = True
            return CoverageResult(
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                src_token_scores=None,
                src_tokens=None,
                subword_agg=subword_agg,
                temperature=temperature,
                is_multi_sentence_input=is_multi_sentence_input,
            )

        tokenizer = self.backward_evaluator.pipeline.tokenizer
        model = self.backward_evaluator.pipeline.model
        clm_tokenizer = self.forward_evaluator.tokenizer
        clm_model = self.forward_evaluator.model
        from translation_models.mbart_models import get_mbart_language
        tokenizer.src_lang = get_mbart_language(self.tgt_lang)
        tokenizer.tgt_lang = get_mbart_language(self.src_lang)

        with torch.no_grad():
            bpe2word_map = get_bpe_map(src, tokenizer)
            inputs = tokenizer._encode_plus(translation, return_tensors="pt").to(device=0)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer._encode_plus(src, return_tensors="pt").to(device=0)
            inputs["decoder_input_ids"] = shift_tokens_right(labels["input_ids"], model.config.pad_token_id)
            output = model(**inputs)
            # get conditional log probabilities
            loss = torch.nn.CrossEntropyLoss(reduction='none')(
                    output.logits[0][1:].view(-1, model.config.vocab_size),
                    labels["input_ids"][0][1:].view(-1),
                    )
            cond_logprob = -loss[:-1]
            w_cond_logprob = get_word_prob(bpe2word_map, cond_logprob, subword_agg=subword_agg)

            # get unconditional log probabilities
            src_tokenised = clm_tokenizer._encode_plus(src, return_tensors="pt").to(device=0)
            src_token_ids = src_tokenised['input_ids'][0].tolist()
            src_tokens = [clm_tokenizer.convert_ids_to_tokens(tok) for tok in src_tokenised['input_ids']][0]
            assert len(src_token_ids) == len(src_tokens)

            clm_output = clm_model(**src_tokenised)
            scal_probs = torch.nn.Softmax(dim=-1)(T_scaling(clm_output['logits'], temperature))
            scal_uncond_prob = torch.empty(size=(len(cond_logprob),)).to(device=0)
            for i in range(scal_probs.shape[1]-2):
                scal_uncond_prob[i] = scal_probs[0][i][src_token_ids[i+1]]

            w_uncond_prob = torch.zeros(size=(len(bpe2word_map),)) 
            import numpy as np
            for i,lst in enumerate(bpe2word_map):
                if len(lst) > 1:
                    if subword_agg == 'product':
                        w_uncond_prob[i] = (torch.prod(torch.tensor([scal_uncond_prob[idx-1] for idx in np.arange(lst[0],lst[-1]+1)])))
                    if subword_agg == 'min':
                        w_uncond_prob[i] = (torch.min(torch.tensor([scal_uncond_prob[idx-1] for idx in np.arange(lst[0],lst[-1]+1)])))
                    if subword_agg == 'first':
                        w_uncond_prob[i] = [scal_uncond_prob[idx-1] for idx in np.arange(lst[0],lst[-1]+1)][0].clone().detach()
                else:
                    w_uncond_prob[i] = scal_uncond_prob[lst[0]-1]

            if self.src_lang == "zh":
                difference = torch.sub(torch.exp(cond_logprob),scal_uncond_prob) 
                score = torch.min(difference)
            else:
                difference = torch.sub(torch.exp(w_cond_logprob),w_uncond_prob) 
                score = torch.min(difference)
            # omission_errors = torch.nonzero((difference == score).float()).view(-1).tolist()

            # if not omission_errors:
            #     omission_errors = []
                
            return CoverageResult(
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                src_token_scores=difference.tolist(),
                src_tokens=src_tokens,
                subword_agg=subword_agg,
                temperature=temperature,
                is_multi_sentence_input=is_multi_sentence_input,
            )