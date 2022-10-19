from typing import List, Tuple
import torch


class TranslationModel:

    def set_language_pair(self, src_lang: str, tgt_lang: str):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

def load_forward_and_backward_model(name: str, src_lang: str, tgt_lang: str, forward_device=0, backward_device=0
                                    ) -> Tuple[TranslationModel, TranslationModel]: #KEEP
    """
    Return the same model twice if it is a many-to-many model, otherwise return two "specialized" models
    """
    from translation_models.mbart_models import load_mbart_one_to_many, load_mbart_causal_lm #PJ
    forward_model = load_mbart_causal_lm(device=0)
    backward_model = load_mbart_one_to_many(tgt_lang, src_lang, device=0) #PJ
    return forward_model, backward_model

# from typing import List


# def batch(input: List, batch_size: int):
#     l = len(input)
#     for ndx in range(0, l, batch_size):
#         yield input[ndx:min(ndx + batch_size, l)]
