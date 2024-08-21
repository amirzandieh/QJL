import torch
from models.llama3_utils_qjl import QJLSketch
from models.llama3_qjl import LlamaForCausalLM_QJL
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_gen_len, **kwargs):
    # 1. prefill stage
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    outputs = model(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
    )
    torch.cuda.synchronize()

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]

    # 2. decoding stage
    torch.cuda.synchronize()
    for i in range(1, max_gen_len):
        outputs = model(
            input_ids=pred_token_idx,
            position_ids=torch.tensor([[input_ids.shape[-1] + len(generated_ids) - 1]], device=input_ids.device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

    del past_key_values
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB

    return peak_memory


def run_test(results):
    for seq_len in seq_lens:
        input_ids = torch.randint(32000, size=(1, seq_len), device='cuda')
        for model_name, model in models.items():
            peak_memory_list = []

            for _ in range(rep):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                peak_memory_ = greedy_generate(model, tokenizer, input_ids, max_gen_len)
                peak_memory_list.append(peak_memory_)

            peak_memory = np.median(peak_memory_list)

            results[model_name]["peak_memory"].append(peak_memory)


def plot_results(tname, filename):
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    rc('font', family='sans-serif', size=20)

    method_names = {'model_exact': 'FP16', 'model_kivi_3': 'KIVI 3-bits', 'model_qjl_3': 'QJL 3-bits',
                    'model_qjl_rht': 'QJL RHT 3-bits', 'model_kivi_5': 'KIVI 5-bits', 'model_qjl_5': 'QJL 5-bits',
                    'model_kvquant': 'KVQuant'}
    colors = [np.array([255, 62, 48]) / 255.0, np.array([23, 107, 239]) / 255.0, '#107C10', '#FFA900']
    markers = ['o', '^', 's', 'D']
    linestyles = ['-', '-.', '-', '--']
    markersizes = [14, 18, 10, 14]
    lw = 7.0
    lsize = 28
    xsize = 38

    seq_lens = [2 ** i for i in range(9, 16)]

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 9))
    i = 0
    for mname in results.keys():
        rr = results[mname]
        ax2.plot(seq_lens, rr[tname], marker=markers[i], linestyle=linestyles[i], color=colors[i],
                 linewidth=lw, markersize=markersizes[i], markeredgecolor='k', markeredgewidth=1.5,
                 label=method_names[mname])
        i += 1

    ax2.tick_params(axis='both', which='major', labelsize=xsize, pad=10)
    ax2.set_xlabel('Sequence Length', size=40, labelpad=10)
    ax2.set_ylabel('Peak Memory (GB)', size=40, labelpad=15)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([2048, 8192, 32000])
    ax2.set_xticklabels(['2k', '8k', '32k'], fontsize=32)
    ax2.grid()
    ax2.legend(bbox_to_anchor=(0, 0.38, 1., .102), ncol=1, fontsize=lsize + 10, framealpha=1,
               edgecolor='k', labelspacing=0.2, borderaxespad=0.5, borderpad=0.3)

    fig.tight_layout(pad=2)
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"
    dtype = torch.bfloat16
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=False,
                                              trust_remote_code=True, )

    config = LlamaConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True

    config.attention_dropout = 0.0
    config.key_quantization_bits = 256
    config.key_quantization_bits_initial_layers = 512
    config.initial_layers_count = 15

    config.outlier_count_general = 0
    config.outlier_count_initial_layers = 0

    config.value_quantization_bits = 2
    config.group_size = 32
    config.buffer_size = 128

    generator = torch.Generator(device=torch.device(device))

    config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=0, rot=True, rng=generator)
    config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=0,
                                          rot=True,
                                          rng=generator)

    config.use_flash = True
    model_qjl = LlamaForCausalLM_QJL(config=config).to(device='cuda', dtype=dtype)

    config = LlamaConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    config.attn_implementation = "flash_attention_2"
    model_exact = LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=dtype).to(device='cuda',
                                                                                                    dtype=dtype)

    seq_lens = [2 ** i for i in range(9, 17)]
    max_gen_len = 64

    models = {
        "model_exact": model_exact,
        "model_qjl_3": model_qjl,
    }

    results = {model_name: {"peak_memory": []} for model_name in models.keys()}
    rep = 10

    run_test(results)
    plot_results('tim_total', 'fig_quant_memory_peak_llama3.png')
