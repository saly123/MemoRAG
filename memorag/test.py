# -*- encoding:utf-8 -*-
# 测试llm的kv获取
from memorag import MemoRAG, Memory, Model

if __name__ == '__main__':
    model_name_or_path = "/data/share/ll_dev/Qwen/Qwen2___5-72B-Instruct"
    memorag = MemoRAG(gen_model_name_or_path=model_name_or_path, memo_model_name_or_path=model_name_or_path,
                      retrival_model_name_or_path=model_name_or_path,
                      retrival_chunk_size=400)



    print("test")
