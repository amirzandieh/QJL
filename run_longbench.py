import os
import argparse
import random
import time
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset
from eval_long_bench import dataset2metric
from fastchat.model import get_conversation_template
from models.llama2_utils_qjl import QJLSketch
from models.llama2_qjl import LlamaForCausalLM_QJL


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(prompt, model_name):
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "longchat" in model_name or "vicuna" in model_name:
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        raise NotImplementedError
    return prompt


def setup_model_and_tokenizer(
        model_name,
        dtype=torch.float16,
        device='cuda',
        attention_dropout=0.0,
        key_quantization_bits=256,
        key_quantization_bits_initial_layers=512,
        initial_layers_count=15,
        outlier_count_general=8,
        outlier_count_initial_layers=8,
        value_quantization_bits=2,
        group_size=32,
        buffer_size=128,
):
    config = LlamaConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        tokenizer_type='llama'
    )

    config = LlamaConfig.from_pretrained(model_name)
    config.attention_dropout = attention_dropout
    config.key_quantization_bits = key_quantization_bits
    config.key_quantization_bits_initial_layers = key_quantization_bits_initial_layers
    config.initial_layers_count = initial_layers_count

    config.outlier_count_general = outlier_count_general
    config.outlier_count_initial_layers = outlier_count_initial_layers

    config.value_quantization_bits = value_quantization_bits
    config.group_size = group_size
    config.buffer_size = buffer_size

    generator = torch.Generator(device=torch.device(device))

    config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
    config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128,
                                              rot=True,
                                              rng=generator)

    config.use_flash = True

    model_qjl = LlamaForCausalLM_QJL.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        cache_dir=None,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    return model_qjl, tokenizer


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="lmsys/longchat-7b-v1.5-32k")
    parser.add_argument('--dtype', type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--attention_dropout', type=float, default=0.0)
    parser.add_argument('--key_quantization_bits', type=int, default=256)
    parser.add_argument('--key_quantization_bits_initial_layers', type=int, default=512)
    parser.add_argument('--initial_layers_count', type=int, default=15)
    parser.add_argument('--outlier_count_general', type=int, default=8)
    parser.add_argument('--outlier_count_initial_layers', type=int, default=8)
    parser.add_argument('--value_quantization_bits', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--n_data', type=int, default=150)
    return parser.parse_args(args)


def load_configurations(config_dir):
    with open(os.path.join(config_dir, 'dataset2maxlen.json')) as f:
        dataset2maxlen = json.load(f)
    with open(os.path.join(config_dir, 'dataset2prompt.json')) as f:
        dataset2prompt = json.load(f)
    with open(os.path.join(config_dir, 'model2maxlen.json')) as f:
        model2maxlen = json.load(f)

    return dataset2maxlen, dataset2prompt, model2maxlen


def evaluate_model(
        model_qjl,
        tokenizer,
        dataset_name,
        device,
        dataset2maxlen,
        dataset2prompt,
        model2maxlen,
        n_data=150,
):
    prompt_format = dataset2prompt.get(dataset_name,
                                       "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:")
    max_length = dataset2maxlen.get(dataset_name, 31500)
    max_gen = model2maxlen.get(dataset_name, 64)

    data = load_dataset('THUDM/LongBench', f"{dataset_name}_e", split='test')
    total_score = 0.
    aa = []
    start = time.time()

    for i in tqdm(range(n_data), desc="Evaluating"):
        json_obj = data[i]
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                                "repobench-p"]:
            prompt = build_chat(prompt, model_qjl.config.name_or_path)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        output = model_qjl.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        ground_truths = json_obj['answers']
        all_classes = json_obj['all_classes']
        prediction = pred

        score = 0.
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset_name](prediction, ground_truth, all_classes=all_classes))

        total_score += score

        mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        mem_reserve = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
        mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak'] / 1024 / 1024 / 1024

        mem_info = f"mem_alloc: {mem_alloc:.5f}, mem_reserved: {mem_reserve:.5f}, mem_peak: {mem_peak:.5f}"
        aa.append(score)
        print(f"[{i:>4}] score: {score:.4f}, avg_score: {total_score / (i + 1):.4f}, | {mem_info}")

    print(f"Average score for dataset {dataset_name}: {np.mean(aa)}")
    print(f"Total evaluation time: {time.time() - start:.2f} seconds")


def main(args):
    seed_everything(args.seed)
    dataset2maxlen, dataset2prompt, model2maxlen,  = load_configurations(args.config_dir)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model_qjl, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        dtype,
        args.device,
        args.attention_dropout,
        args.key_quantization_bits,
        args.key_quantization_bits_initial_layers,
        args.initial_layers_count,
        args.outlier_count_general,
        args.outlier_count_initial_layers,
        args.value_quantization_bits,
        args.group_size,
        args.buffer_size,
    )
    print(f"Model and tokenizer for {args.model_name} are set up successfully.")
    evaluate_model(
        model_qjl,
        tokenizer,
        args.dataset_name,
        args.device,
        dataset2maxlen,
        dataset2prompt,
        model2maxlen,
        args.n_data,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
