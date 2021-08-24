import sentencepiece as spm
import torch
import os

if __name__ == '__main__':
    s1 = spm.SentencePieceProcessor(model_file='mt5_model/mt5-base/spiece.model')
    s2 = spm.SentencePieceProcessor(model_file="mt5_model/mt5-base-simplify/spiece_cn.model")
    print(s1.Encode("我"))
    print(s2.Encode("我"))

    source_state_orderdict = torch.load(os.path.join("mt5_model/mt5-base", "pytorch_model.bin"))
    simplify_state_orderdict = torch.load(os.path.join("mt5_model/mt5-base-simplify", "pytorch_model.bin"))
    print(simplify_state_orderdict.__sizeof__())
    s1_e = source_state_orderdict["shared.weight"]
    s2_e = simplify_state_orderdict["shared.weight"]

    print(s1_e[3003] == s2_e[1182])
    # print(s2_e[1475])