from dataclasses import dataclass, field
import json
import os
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm
from transformers import TrainerCallback
from transformers import Trainer, deepspeed
# 设置忽略的标签ID
IGNORE_TOKEN_ID = -100

# 禁用W&B日志记录
os.environ["WANDB_MODE"] = "disabled"

# 加载处理器和分词器
min_pixels = 64 * 28 * 28
max_pixels = 128 * 28 * 28

processor = AutoProcessor.from_pretrained(
    "./Qwen-VL/model/processor",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    padding_side="left"
)
processor.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

tokenizer = AutoTokenizer.from_pretrained("./Qwen-VL/model/tokenizer", trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id

def load_json_file(path):
    """加载 JSON 文件并返回数据列表。"""
    if not os.path.isfile(path):
        raise ValueError(f"路径 {path} 不是一个文件")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 过滤掉空的或无效的条目
    data = [entry for entry in data if entry]
    return data

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")

@dataclass
class DataArguments:
    data_path: str = field(
        default="/path/to/your/training/data", metadata={"help": "训练数据的路径。"}
    )
    eval_data_path: str = field(
        default="/path/to/your/evaluation/data", metadata={"help": "评估数据的路径。"}
    )
    lazy_preprocess: bool = True

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度。序列将被右填充（或截断）。"}
    )
    fix_vit: bool = True
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    output_dir: str = field(default="/path/to/save/model")
    deepspeed: str = field(default="finetune/ds_config_zero3.json")
    bf16: bool = True  # 启用混合精度训练
    remove_unused_columns: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    warmup_ratio: float = field(default=0.06)
    weight_decay: float = field(default=0.01)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """收集状态字典并保存到磁盘。"""
    # 检查是否启用了Zero-3模式
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

class PreprocessedDataset(Dataset):
    """使用预处理的数据集。"""

    def __init__(self, data):
        super(PreprocessedDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.data[idx]:
            raise ValueError(f"在索引 {idx} 处的数据为空")
        return self.data[idx]

def find_assistant_content_sublist_indexes(token_ids):
    """找到assistant内容在token_ids中的起始和结束索引。"""
    assistant_start_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    start_indexes = []
    end_indexes = []

    # 寻找assistant内容的起始和结束位置
    for i in range(len(token_ids) - len(assistant_start_ids) + 1):
        if token_ids[i:i+len(assistant_start_ids)] == assistant_start_ids:
            start_indexes.append(i)
            for j in range(i + len(assistant_start_ids), len(token_ids)):
                if token_ids[j] == im_end_id:
                    end_indexes.append(j)
                    break
    return list(zip(start_indexes, end_indexes))


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    # 设置配置
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # 加载模型和分词器
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        config=config
    )
    print("load model from " + model_args.model_name_or_path)

    # 冻结视觉模型的参数
    if training_args.fix_vit and hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
        model.transformer.visual.requires_grad_(False)
        if hasattr(model.transformer.visual, 'attn_pool'):
            model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = AutoTokenizer.from_pretrained("./Qwen-VL/model/tokenizer", trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id

    # 加载训练数据
    train_data = load_json_file(data_args.data_path)
    train_dataset = PreprocessedDataset(train_data)

    # 加载评估数据
    if data_args.eval_data_path:
        eval_data = load_json_file(data_args.eval_data_path)
        eval_dataset = PreprocessedDataset(eval_data)
    else:
        eval_dataset = None

    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

    # 定义collate_fn函数
    def collate_fn(batch):
        # batch 是一个列表，每个元素是一个对话（消息列表）
        custom_system_prompt = "You are an AI assistant specialized in emotion recognition. For all questions, you only need to choose one word from Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise to reply. No need to reply with other words."
        
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, system=custom_system_prompt)
            for msg in batch
        ]

        # 从 messages 中排除 assistant 的回复
        # messages_without_assistant = []
        # for msg in batch:
        #     msg_without_assistant = [m for m in msg if m['role'] != 'assistant']
        #     messages_without_assistant.append(msg_without_assistant)

        # print("texts:")
        # print(texts)
        # 处理图像和视频输入
        image_inputs, video_inputs = process_vision_info(batch)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=training_args.model_max_length,
            return_tensors="pt",
        )

        input_ids_lists = inputs['input_ids'].tolist()
        assert len(batch) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [IGNORE_TOKEN_ID] * len(ids_list)
            for begin_end_indexes in find_assistant_content_sublist_indexes(ids_list):
                # 确保索引不超过序列长度
                start = begin_end_indexes[0] + 2
                end = begin_end_indexes[1] + 1
                start = min(start, len(ids_list))
                end = min(end, len(ids_list))
                label_ids[start:end] = ids_list[start:end]
            labels_list.append(label_ids)

        labels_ids = torch.tensor(labels_list, dtype=torch.int64)

        return dict(
            input_ids=inputs['input_ids'],
            labels=labels_ids,
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs['image_grid_thw']
        )

    # 初始化自定义的 Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        data_collator=collate_fn,
        # callbacks=[OutputCallback(print_every_n_steps=10)]  # 添加回调
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
