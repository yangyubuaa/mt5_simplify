import sentencepiece as spm

if __name__ == '__main__':
    s = spm.SentencePieceProcessor(model_file='mt5_model/mt5-base-simplify/spiece.model')
    print(s.Encode("你好啊"))
    # [259, 1475, 1318, 6406]