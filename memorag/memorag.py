# -*- encoding:utf-8 -*-
# author: linlizhai
# memory model  + infer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
# from semantic_text_splitter import TextSplitter
import os
from transformers.tokenization_utils_base import BatchEncoding


class Model:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left")

    def input2ids(self, inputs):
        # 两种方式（方式一）
        format_encode_inputs = self.tokenizer.apply_chat_template(inputs, tokenize=False,
                                                                  add_generation_prompt=False)  # add_generation_prompt =True，format之后会增加一个 assistant的内容
        encode_inputids = self.tokenizer(format_encode_inputs, add_special_tokens=False, return_tensors="pt",
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = None
        self.context = None
        self.memo_type = "longllm"  # longllm和beacon两种（具体怎么区分待研究）——加载和存储memory的格式不一样

    def memorize(self, context):
        '''
        :param context: context生成kv 保存
        :return:
        '''
        context_system = f'''
        You are provided with a long article. Read the article carefully. After reading, you will be asked to perform specific tasks based on the content of the article.

Now, the article begins:
- **Article Content:** {context}

The article ends here.

Next, follow the instructions provided to complete the tasks.
        '''
        context_prompt = [{"role": "user", "content": context_system},
                          {"role": "assistant", "content": "I have read the article. Please provide your question."}]

        format_contextids = self.input2ids(context_prompt)
        self.memory = DynamicCache()
        with torch.no_grad():
            self.model(**format_contextids, past_key_values=self.memory)
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

    def generate(self, query, max_new_tokens=512, do_sample=False, temperature=0, with_cache=True):
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

        query_input = [{"role": "system",
                        "content": "你是携程的酒店商服人员，你的服务对象是入驻携程平台的酒店商户，你的职责是回答商户关于酒店经营的问题。请你阅读“资料”，结合给定的意图层级结构回答“问题”。\n回答问题时必须严格遵守如下要求：\n(1)严格根据提取的原始片段内容生成答案，尽量从片段中摘取能回答该问题的专业词汇，不可以出现资料中没有的内容。\n(2)在资料信息不足以回答问题或者资料与问题不相关时，你必须直接回答“抱歉，经过检索我没有找到这个问题的答案。”\n(3)如果资料中没有正面直接的回答问题，你必须直接回答“抱歉，经过检索我没有找到这个问题的答案。”。\n(4)对于<richText><description>DESCRIPTION</description><link>LINK</link></richText>格式的标签，如需引用仅需输出LINK即可，无需输出DESCRIPTION文本。\n(5)如果你在回答中让用户参考图片，那么你就必须输出图片LINK。\n(6)回答问题时语言要简洁，语句要通顺并且具有逻辑性，不可以出现语义重复，例如：不能出现“加入xx加盟”、“根据xx”类似的表达。\n(7)回答的开头需要结合问题进行改写输出，但是不要直接将意图结构输出，也不要出现类似“子节点意图”类似表述。回答内容必须与问题直接相关，不要出现问题没有提到的内容。\n(8)必须按照给定的回答角度和能提取出相关内容的意图层级结构形成答案，如果资料中没有直接提到某个子意图，则该子意图一定不要输出。\n(9)不要出现请咨询客服等话术，不要出现重复的内容和多个连续的“\n”字符，生成的回答必须与问题相关。\n(10)如果回答的内容在资料中有类似<richText></richText>的操作内容，请正确引用并输出对应的LINK文本。\n(11)请直接输出最终的回答，回答的开头一定不要出现类似：“关于xx”、“我可以从xx”、“虽然资料中没有直接提到xx”、“根据资料xx”、“xx提供的资料xx”的句式，请直接正面回答问题。请严格按照前面的要求输出回答，不需要输出对资料的总结。\n(12)请严格按照给定的意图结构回答，不要编造新的结构，如果某个意图要点在资料中找不到相关内容解释，请一定不要只输出该意图。输出的每个意图需要一定的简单阐述，最终的输出应该尽量简洁。输出格式尽量紧凑，单独的小要点尽量合并。\n(13)请一定不要输出资料中没有出现的超链接，也不要编造<richText><link>LINK</link></richText>格式、<richText><link>https://example.com/</link></richText>格式、https://example.com/格式的输出。\n(14)如果资料足以回答问题，请一定不要在回答中混合输出“抱歉，经过检索我没有找到这个问题的答案。”，也一定不要在回答中混合输出“具体内容未详细提及”。\n(15)回答中一定不要出现类似“https://example.com/”的内容。\n(16)回答多个问题时，请将原来的问题修改为符合语法逻辑的句子，多个问题中，如果某个问题的回答是”抱歉，经过检索我没有找到这个问题的答案。“，请一定不要输出该问题。\n(17)输出的内容一定不要出现重复。\n(18)错误的输出格式1：【根据问题，儿童政策可以设置的种类如下：\n1. 是否允许儿童入住。抱歉，经过检索我没有找到这个问题的答案】。错误的输出格式2：【云梯：抱歉，经过检索我没有找到这个问题的答案】。错误的输出格式3：【psi奖励分：资料中未提及】。请一定不要出现类似前面的错误格式。\n(19)输出的要点请一定不要带有“抱歉，经过检索我没有找到这个问题的答案”的内容。"},
                       {"role": "user",
                        "content": "parent_intent表示父节点意图，child_intent表示子节点意图。付费的意图如下：【列表页营销活动、携程站外营销活动、钻石展位】，免费的意图如下：【委托分销、PSI、活动及权益、额外渠道流量、旅拍文章、预售产品（常规预售+时令直播 如919）】，给定具体的意图层级结构：【[{\"parent_intent\":\"产量\",\"child_intent\":[\"流量\",\"转化\"]},{\"parent_intent\":\"流量\",\"child_intent\":[\"列表页流量\",\"额外流量\"]},{\"parent_intent\":\"转化\",\"child_intent\":[\"列表页转化率\",\"详情页转化率\",\"订单页转化率\"]},{\"parent_intent\":\"列表页流量\",\"child_intent\":[\"列表页营销活动\",\"委托分销\",\"PSI\"]},{\"parent_intent\":\"额外流量\",\"child_intent\":[\"活动及权益\",\"携程APP首页流量\",\"额外渠道流量\",\"携程站外营销活动\"]},{\"parent_intent\":\"列表页转化率\",\"child_intent\":[\"优化列表转化的基础信息\",\"优化列表转化的活动标签\",\"优化列表转化的权益标签\",\"价格类\"]},{\"parent_intent\":\"详情页转化率\",\"child_intent\":[\"详情页的基础信息\",\"详情页的房型信息\",\"详情页的点评\"]},{\"parent_intent\":\"订单页转化率\",\"child_intent\":[\"权益\",\"支付\"]},{\"parent_intent\":\"列表页营销活动\",\"child_intent\":[\"金字塔广告\",\"竞争圈定投\",\"云梯\"]},{\"parent_intent\":\"委托分销\",\"child_intent\":[\"二级委托分销\",\"一级委托分销\",\"履约情况（如果委托分销期间未履约，会影响流量）\"]},{\"parent_intent\":\"PSI\",\"child_intent\":[\"psi基础分（产量、价格等是否得分）\",\"psi奖励分\",\"psi扣分项（申诉或整改）\"]},{\"parent_intent\":\"活动及权益\",\"child_intent\":[\"提前入住\",\"时令活动\",\"联名会员\",\"积分联盟\",\"成长任务（完成相关激励活动）\",\"免费升级\",\"免费兑换早餐\",\"优享会\",\"pkg出行特惠\",\"提前预订促销\",\"连住特惠\",\"延迟退房\"]},{\"parent_intent\":\"携程APP首页流量\",\"child_intent\":[\"旅拍文章\",\"钻石展位\",\"预售产品（常规预售+时令直播 如919）\"]},{\"parent_intent\":\"额外渠道流量\",\"child_intent\":[\"商旅加盟\"]},{\"parent_intent\":\"携程站外营销活动\",\"child_intent\":[\"酒店新媒体推广代运营\",\"外部媒体广告\",\"星力引擎\"]},{\"parent_intent\":\"优化列表转化的基础信息\",\"child_intent\":[\"首图设置\",\"列表页7秒小视频\",\"长标签\",\"短标签\",\"点评分\"]},{\"parent_intent\":\"优化列表转化的活动标签\",\"child_intent\":[\"积分联盟\",\"闪住\",\"优享会\"]},{\"parent_intent\":\"优化列表转化的权益标签\",\"child_intent\":[\"免费升级\",\"免费兑换早餐\",\"延迟退房\"]},{\"parent_intent\":\"价格类\",\"child_intent\":[\"起价\"]},{\"parent_intent\":\"详情页的基础信息\",\"child_intent\":[\"开业/装修时间维护\",\"亮点信息维护\",\"图片\",\"用户推荐词\",\"标签描述维护\",\"酒店图文\",\"酒店简介\",\"设施设备信息\",\"订房必读\",\"儿童政策\",\"详情页酒店视频（90s）\",\"酒店入离政策\"]},{\"parent_intent\":\"详情页的房型信息\",\"child_intent\":[\"无早、单早、双早房型覆盖\",\"房型照片/视频\",\"房型价格梯度\",\"现预付覆盖\",\"房型名称\",\"入住既享礼盒\",\"日历房套餐房型\",\"门票房型\",\"房型及时确认\",\"房型取消政策\",\"钟点房\"]},{\"parent_intent\":\"详情页的点评\",\"child_intent\":[\"一级委托分销置顶点评权益\",\"点评回复\"]},{\"parent_intent\":\"权益\",\"child_intent\":[\"积分联盟\",\"免费升级\",\"免费兑换早餐\"]},{\"parent_intent\":\"支付\",\"child_intent\":[\"闪住\",\"分期付款（参与委托分销酒店可享）\"]}]】。请结合意图结构从【流量、转化】角度，回答下面的问题：【" + query + "】。回答多个问题时不要出现内容的重复。请先按照意图结构逐个总结子节点意图相关的内容，不可以遗漏任何子节点意图在资料中出现的相关内容，如果该子意图找不到相关的片段，则直接删除该意图结构。再根据提取的片段回答问题。你生成的答案需要满足下面的条件：1、严格根据提取的原始片段内容生成答案，尽量从片段中摘取能回答该问题的专业词汇，不可以出现资料中没有的内容；2、按照回答角度和能提取出相关内容的意图层级结构形成答案，如果该意图在资料中找不到相关内容，则该意图结构可以在答案中不输出；3、语言要简洁，语句要通顺并且具有逻辑性，不要出现表达重复的语句；4、不要出现请咨询客服等话术；5、生成的回答有逻辑层次结构，不要出现重复的内容。请直接输出最终的回答，回答的开头需要与问题相结合。根据这个问题，我们检索到如下相关资料：【相关的名词解释如下：【名词】：促销，【解释】：携程推出的活动，非付费的营销活动，也不是权益。【名词】：金字塔，【解释】：携程推出的一种付费广告服务。【名词】：psi，【解释】：psi分代表酒店的综合服务质量，直接影响酒店在携程平台上的流量竞争力。【名词】：委托分销，【解释】：是携程的头部商家合作计划。【名词】：权益，【解释】：携程推出的一项服务型功能，酒店可以通过加入权益云计划，提供不同的权益给用户，权益具体包括以下内容：免费升级、免费兑换早餐、延迟退房、提前入住。权益不是一种促销活动。【名词】：出行特惠，【解释】：对预订过机票、火车票、门票等出行产品的用户，在预订酒店的时候可以享受部分特惠的折扣。是一个单独的活动，出行特惠不是权益。【名词】：流量快车，【解释】：流量快车（CPC）是同程旅行带客宝推出的通过帮助酒店提升曝光，引入潜在客户的付费推 广产品，免费曝光，点击后计费，通过对优质流量资源的高效配置，利用专业数据处理算法实现成本可控、效益可观，精准定位的推广投放系统。【名词】：流量加油站，【解释】：流量加油站（CPM）是同程旅行带客宝推出的以自然排序的形式，无广告标识，帮助酒店获 取更多的展示机会，提高曝光，引入潜在客户，通过对优质流量资源的高效配置，利用专业数据处理算法实现成本可控、效益可观，精准定位的广告投放产品。【名词】：亮点信息，【解释】：亮点信息内容包含酒店图文、标签和旅拍等。【名词】：现预付覆盖，【解释】：酒店现预付房型均需设置，以满足客人需求。】。请直接输出最终的回答，回答的开头需要与问题相结合，答案一定不要出现类似：“我可以从xx”、“虽然资料中没有直接提到xx”的句式。"}]
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


class MemoRAG():
    def __init__(self, gen_model_name_or_path, memo_model_name_or_path, retrival_model_name_or_path,
                 retrival_chunk_size=400):
        self.gen_model_name_or_path = gen_model_name_or_path
        self.memo_model_name_or_path = memo_model_name_or_path
        self.retrival_model_name_or_path = retrival_model_name_or_path
        self.retrival_chunk_size = retrival_chunk_size

        self.gen_model = Model(self.gen_model_name_or_path)
        self.memo_model = Memory(self.memo_model_name_or_path)

        self.retrival_model = None
        # self.text_spliter = TextSplitter.from_tiktoken_model(
        #     "gpt-3.5-turbo", self.retrival_chunk_size)

        self.text_spliter = None

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
