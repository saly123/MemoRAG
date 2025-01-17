# -*- encoding:utf-8 -*-
# memory model  + infer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
from semantic_text_splitter import TextSplitter
import os
from transformers.tokenization_utils_base import BatchEncoding


class Model:
    def __init__(self):
        self.model_name_or_path = ""
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left")

    def input2ids(self, inputs):
        # 两种方式（方式一）
        format_encode_inputs = self.tokenizer.apply_chat_template(inputs, tokenize=False,
                                                                  add_generation_prompt=False)  # add_generation_prompt =True，format之后会增加一个 assistant的内容
        encode_inputids = self.tokenizer.tokenizer(format_encode_inputs, add_special_tokens=False, return_tensors="pt",
                                                   padding=True)  # padding_side = "left"

        # # 两种方式（方式二）
        # encode_inputids = self.tokenizer.apply_chat_template(inputs, tokenize=True, add_generation_prompt=False)
        return encode_inputids

    def id2text(self, outputids):
        outputs = self.tokenizer.batch_decode(outputids, skip_special_tokens=True)
        return outputs

    def generate(self, inputs, **generation_kwargs):
        '''
        :param inputs: batch inputs
        :return:
        '''

        encode_inputids = self.input2ids(inputs)

        outputids = self.model.generate(encode_inputids["input_ids"], attention_mask=encode_inputids["attention_mask"],
                                        **generation_kwargs)

        outputs = self.id2text(outputids)

        return outputs


class Memory(Model):
    def __init__(self):
        super().__init__()
        self.memory = None
        self.context = None
        self.memo_type = "longllm"  # longllm和beacon两种（具体怎么区分待研究）——加载和存储memory的格式不一样

    def memorize(self, context):
        '''
        :param context: context生成kv 保存
        :return:
        '''

        format_contextids = self.input2ids(context)
        self.memory = DynamicCache()
        with torch.no_grad():
            self.model(**format_contextids, past_key_value=self.memory)
        self.memory = self.model.past_key_values
        self.context = format_contextids

    def save(self, path):
        '''
        memory past_key_value保存
        :param path:
        :return:
        '''
        torch.save({"memory": self.memory, "context": self.context}, path)

    def load(self, path):
        '''
        加载memory模型
        :param path:
        :return:
        '''
        _cache = torch.load(path)
        self.memory = _cache["memory"]
        self.context = _cache["context"]

    def merge_input(self, input1, input2):
        '''
        :param input1: 来自tokenizer进行tokenize之后的结果，包含两个key（input_ids和attention_mask）
        :param input2:
        :return:
        '''
        merged_inputids = torch.concat([input1["input_ids"], input2["input_ids"]], dim=1)
        merged_attention_mask = torch.concat([input1["attention_mask"], input2["attention_mask"]], dim=1)
        merged_input = BatchEncoding({"input_ids": merged_inputids, "attention_mask": merged_attention_mask})
        return merged_input

    def generate(self, query, prompt_template, max_new_tokens=512, do_sample=False, temperature=0, with_cache=True):
        '''
        结合memory生成回答
        :param query:
        :param prompt_template:
        :param max_new_tokens:
        :param with_cache: 是否使用memory缓存
        :return:
        '''

        # 输入可能是多个instruct

        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "temperature": temperature}
        if self.memo_type == "longllm" and with_cache:
            generation_kwargs["past_key_value"] = self.memory  # copy.deepcopy()???

        query_input = [{"role": "system", "content": "xxxx"}, {"role": "user", "content": "xxxx"}]
        query_inputids = self.input2ids(query_input)

        # 将memory的context和当前的query 拼接后用model generate
        sample_inputs = self.merge_input(self.context, query_inputids)
        response = self.id2text(sample_inputs)

        if self.memo_type == "longllm" and with_cache:
            # 推理结束后删除memory缓存
            del generation_kwargs["past_key_value"]
            torch.cuda.empty_cache()

        return response

    def recall(self, query):
        return

    def answer(self, query, max_new_tokens=512):
        # query 需要拼接成prompt的格式
        return self.model.generate(query, max_new_tokens=max_new_tokens)

    def rewrite(self, query):
        return


class MemoRAG(Model):
    def __init__(self, gen_model_name_or_path, memo_model_name_or_path, retrival_model_name_or_path,
                 retrival_chunk_size=400):
        super().__init__()
        self.gen_model_name_or_path = gen_model_name_or_path
        self.memo_model_name_or_path = memo_model_name_or_path
        self.retrival_model_name_or_path = retrival_model_name_or_path
        self.retrival_chunk_size = retrival_chunk_size

        self.gen_model = Model(self.gen_model_name_or_path)
        self.memo_model = Memory(self.memo_model_name_or_path)

        self.retrival_model = None
        self.text_spliter = TextSplitter.from_tiktoken_model(
            "gpt-3.5-turbo", self.retrival_chunk_size)

        self.corpus_chunk = None

    def memorize(self, context, save_dir):
        self.memo_model.memorize(context)
        self.corpus_chunk = self.text_spliter.split_text(context)
        self.retrival_model.add(self.corpus_chunk)  # 加入检索库

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            self.memo_model.save(os.path.join(save_dir, "memory.bin"))
            # index save

    def load(self, save_dir):
        self.memo_model.load(os.path.join(save_dir, "memory.bin"))

    def _prepare_retrieval_query(self, query, use_memory_answer=True):
        '''
        :param query:
        :param use_memory_answer:
        :return:
        '''
        retrieval_chunk = self.retrival(query)  # query检索相似chunk
        potential_answer = None  # query仅根据context生成的answer，补充chunk

        # 可以将只用context生成的answer作为部分chunk
        if use_memory_answer:
            potential_answer = self.memo_model.answer(query)
            retrieval_chunk.append(potential_answer)
        return retrieval_chunk, potential_answer

    def retrival(self, query, hits=10):
        # 根据query召回相关chunk
        # 检索之前可以先对query进行改写，这样能提高检索的准召回率
        return []

    def handle_rag(self, query, prompt_template, user_memory_answer=False, max_new_tokens=512):
        retrival_chunk, potential_answer = self._prepare_retrieval_query(query,
                                                                         use_memory_answer=True)  # 可以不借助chunk，直接用记忆的context进行回答

        # chunk拼接
        response = self._generate_response(query, retrival_chunk, max_new_tokens=1024)

        return response

    def _generate_response(self, query, knowledge, max_new_tokens=512):
        '''
        :param query: query
        :param knowledge: knowledge【past key value】
        :param max_new_tokens: max_new_tokens
        :return:
        '''
        # knowledge 拼接到prompt
        rag_system_prompt = ""
        prompt = [{"role": "system", "content": rag_system_prompt + "相关资料" + knowledge},
                  {"role": "user", "content": "当前输入的问题：" + query}]

        # 结合cache
        if self.gen_model.__class__.__name__ == "Memory":
            output = self.gen_model.generate(prompt, max_new_tokens=max_new_tokens, with_cache=True)[0]
        elif self.gen_model.__class__.__name__ == "Model":
            # model没有cache
            output = self.gen_model.generate(prompt, max_new_tokens=max_new_tokens)[0]

        return output
