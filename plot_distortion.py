import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from models.llama2_utils_qjl import QJLSketch, QJLKeyQuantizer
import argparse


def calculate_error(query, key, key_quantization_bits, outlier_count, buffer_size):
    actual = torch.matmul(query, key.transpose(-1, -2))

    generator = torch.Generator(device=torch.device('cuda'))
    qjl = QJLSketch(dim=(128, key_quantization_bits), dim_outlier=key_quantization_bits, rot=True, rht=False, rng=generator)
    k_quant = QJLKeyQuantizer(qjl, outlier_count, buffer_size, 32, key_quantization_bits)
    k_quant.build_sketch(key)

    att = k_quant.attention_score(query)
    error = torch.linalg.norm(att - actual) / torch.linalg.norm(actual)
    return error.item()


def main(args):
    layers = [0, 1, 2, 4, 8, 16, 31]
    results_dict = {}

    for layer in layers:
        key = torch.load(f'{args.key_path}/prompt_key-layer={layer}.pt')[:, :, :, :].to(torch.bfloat16)
        query = torch.load(f'{args.query_path}/prompt_query-layer={layer}.pt')[:, :, -1:, :].to(torch.bfloat16)

        key_quantization_bits_values = [64 * i for i in range(1, 9)]
        top_coord = 0

        errors = np.zeros((len(key_quantization_bits_values)))
        num_bits = np.zeros((len(key_quantization_bits_values)))
        rep = 1
        for j, key_quantization_bits in enumerate(key_quantization_bits_values):
            for _ in range(rep):
                error = calculate_error(query.clone(), key.clone(), key_quantization_bits, top_coord, 128)
                errors[j] += error / rep
                num_bits[j] = key_quantization_bits / 64

        results_dict[layer] = {
            'errors': errors,
            'num_bits': num_bits
        }

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    rc('font', family='sans-serif', size=20)

    layer_names = {0: 'Layer 0', 1: 'Layer 1', 2: 'Layer 2', 4: 'Layer 4', 8: 'Layer 8', 16: 'Layer 16', 31: 'Layer 31'}
    colors = [np.array([255, 62, 48]) / 255.0, np.array([23, 107, 239]) / 255.0, '#00CED1', '#107C10', '#FFA900', '#8A2BE2',
              '#FF5733', ]
    markers = ['o', '^', 's', 'D', 'v', 'x', 'p']
    linestyles = ['-', '-.', '-', '--', ':', '-.', '--']
    markersizes = [14, 18, 10, 14, 12, 16, 12]
    lw = 3.0
    lsize = 28
    xsize = 24

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    for i, (layer, data) in enumerate(results_dict.items()):
        num_bits = data['num_bits']
        errors = data['errors']
        ax.plot(num_bits, errors, marker=markers[i], linestyle=linestyles[i + 1], color=colors[i],
                linewidth=lw, markersize=markersizes[i], markeredgecolor='k', markeredgewidth=1.5,
                label=f"{layer_names[layer]}")

    ax.tick_params(axis='both', which='major', labelsize=xsize, pad=10)
    ax.set_xlabel(r'Bits per Channel ($\frac{m}{d}$)', size=36, labelpad=10)
    ax.set_ylabel(r'Distortion ($\epsilon$)', size=36, labelpad=15)
    ax.set_yscale('log')
    ax.grid(True)

    ax.legend(bbox_to_anchor=(0.5, 1), fontsize=lsize, framealpha=1,
              edgecolor='k', labelspacing=0.2, borderaxespad=0.5, borderpad=0.3)

    fig.tight_layout(pad=2)
    plt.savefig('error_vs_num_bits.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process key and query paths for quantization error analysis.')
    parser.add_argument('--key_path', type=str, required=True, help='Path to the directory containing key tensors.')
    parser.add_argument('--query_path', type=str, required=True, help='Path to the directory containing query tensors.')
    args = parser.parse_args()

    main(args)
