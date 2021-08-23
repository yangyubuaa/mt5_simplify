import yaml
import torch
import os
import json
from collections import OrderedDict
from mt5_model.transformers_pkg.models.t5 import T5Model


def simp(config):
    source_path = config["source_path"]
    target_path = config["target_path"]

    # 加载原始参数到source_state_orderdict
    source_state_orderdict = torch.load(os.path.join(source_path, "pytorch_model.bin"))

    # print(source_state_orderdict)

    # 得到shared参数（即embedding参数，encoder和decoder共享）
    shared = source_state_orderdict["shared.weight"]
    print(shared.shape)  # (250112, 768) 我们可以计算出shared层占用的参数量为250112 * 768 = 192M
    # 占用的内存为768MB左右，所以需要进行精简
    # 读取id，假设id为[0, 2]
    pass
    simplified = torch.index_select(shared, 0, torch.tensor([0, 2]))
    print(simplified.shape)
    new_embedding_params = OrderedDict()
    new_embedding_params["shared.weight"] = simplified
    new_embedding_params["encoder.embed_tokens.weight"] = simplified
    new_embedding_params["decoder.embed_tokens.weight"] = simplified
    new_embedding_params["lm_head.weight"] = simplified

    # 更新source_state_dict参数
    source_state_orderdict.update(new_embedding_params)
    torch.save(source_state_orderdict, target_path)
    # with open(target_path, "w") as w:
    #     json.dump(source_state_orderdict, w)


def verify(config):
    m = T5Model.from_pretrained(config["target_path"])


if __name__ == '__main__':
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    # simp(config)
    verify(config)