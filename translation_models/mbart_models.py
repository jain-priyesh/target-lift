# from lib2to3.pgen2.tokenize import tokenize
# from typing import List

import os #PJ
# import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Text2TextGenerationPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM #PJ
# from transformers.file_utils import PaddingStrategy
# from transformers.models.mbart.modeling_mbart import shift_tokens_right
# from transformers import logging as log2#PJ
# log2.set_verbosity_error() #PJ
# import logging #PJ

from translation_models import TranslationModel
# from translation_models.utils import batch

class MbartCausalLMModel():
    
    def __init__(self,
                 model_name_or_path: str = "facebook/mbart-large-50",
                 device=None,
                 *args,
                 **kwargs
                 ): #PJ
        if not os.path.isdir(model_name_or_path):
            model_name_or_path: str = "facebook/mbart-large-50"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model =  AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        self.model_name_or_path = model_name_or_path
        self.mask_id = self.tokenizer.convert_tokens_to_ids('<mask>')


    def __str__(self): #PJ
        return self.model_name_or_path.replace("/", "_")

class Mbart50Model(TranslationModel):

    def __init__(self,
                 model_name_or_path: str = "facebook/mbart-large-50-one-to-many-mmt",
                 device=None,
                 *args,
                 **kwargs
                 ):
        
        if not os.path.isdir(model_name_or_path): #PJ
            model_name_or_path: str = "facebook/mbart-large-50-one-to-many-mmt" 
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.pipeline = Text2TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=device,
            *args, **kwargs,
        )

    def __str__(self):
        return self.model_name_or_path.replace("/", "_")

    @property
    def is_to_many_model(self):
        return "-to-many-" in self.model_name_or_path

    def set_language_pair(self, src_lang: str, tgt_lang):
        src_lang = get_mbart_language(src_lang)
        tgt_lang = get_mbart_language(tgt_lang)
        assert not self.is_to_many_model or src_lang == "en_XX"
        assert self.is_to_many_model or tgt_lang == "en_XX"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.pipeline.tokenizer.src_lang = src_lang
        self.pipeline.tokenizer.tgt_lang = tgt_lang

def load_mbart_one_to_many(src_lang: str, tgt_lang: str, device: int = None) -> Mbart50Model: #KEEP
    assert "en" in {src_lang, tgt_lang}
    assert src_lang != tgt_lang
    if src_lang == "en":
        model_name = "/mnt/a99/d0/priyeshjain/models/mbart-large-50-one-to-many-mmt"
    else:
        model_name = "/mnt/a99/d0/priyeshjain/models/mbart-large-50-many-to-one-mmt"
    return Mbart50Model(model_name_or_path=model_name, device=device)
    
def load_mbart_causal_lm(device: int = None): 
    return MbartCausalLMModel(model_name_or_path="/mnt/a99/d0/priyeshjain/models/mbart-large-50", device=device)

def get_mbart_language(language: str):
    mbart_language_codes = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
                            "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
                            "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL",
                            "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF",
                            "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA",
                            "gl_ES", "sl_SI"]
    code_dict = {code.split("_")[0]: code for code in mbart_language_codes}
    if "_" in language:
        assert language in code_dict.values()
        return language
    else:
        return code_dict[language]
