import matplotlib.pyplot as plt
import numpy as np

from simulation import generate_ccd_signal, ADC_MAX
from signal_processing import process_signal
from marker_code import cells_to_bits, bits_to_string, find_marker, extract_codes_after_markers


def plot_debug(processed, markers, codes):
    raw = processed.raw
    lower = processed.lower_envelope
    upper = processed.upper_envelope
    norm = processed.normalized
    thr = processed.local_threshold
    binary = processed.binary
    left, right = processed.roi
    centers = processed.centers
    pitch = processed.pitch

    fig, ax = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

    ax[0].imshow(raw[np.newaxis, :], cmap="gray", aspect="auto",
               vmin=0, vmax=ADC_MAX)

    ax[1].plot(raw, color='black', lw=1.2, label='raw')
    ax[1].plot(lower, '--', lw=1.0, label='lower envelope')
    ax[1].plot(upper, '--', lw=1.0, label='upper envelope')
    ax[1].axvline(left, color='tab:red', ls=':')
    ax[1].axvline(right, color='tab:red', ls=':')
    ax[1].legend()
    ax[1].set_ylabel('ADC')

    ax[2].plot(norm, color='tab:blue', lw=1.2, label='normalized')
    ax[2].plot(thr, color='tab:orange', lw=1.0, label='Bradley threshold')
    edges = (centers[1:] + centers[:-1]) / 2 - pitch
    for e in edges:
        ax[2].axvline(e, color='gray', alpha=0.2)
    ax[2].legend()
    ax[2].set_ylabel('Norm')

    ax[3].step(range(len(binary)), binary, where='mid', color='black', label='binary')
    for e in edges:
        ax[3].axvline(e, color='gray', alpha=0.2)
    for marker in markers:
        x1 = edges[0] + pitch * marker.start_index
        x2 = x1 + pitch * len(marker.matched_bits)
        ax[3].axvspan(x1, x2, alpha=0.3, color='red')
    for code in codes:
        x1 = edges[0] + pitch * code.code_start
        x2 = x1 + pitch * len(code.code_bits)
        ax[3].axvspan(x1, x2, alpha=0.3, color='green')
    ax[3].axvline(left, color='tab:red', ls=':')
    ax[3].axvline(right, color='tab:red', ls=':')
    ax[3].set_ylabel('Bit')
    ax[3].set_xlabel('Pixel')
    ax[3].legend()

    plt.tight_layout()
    plt.show()


def run_demo():
    marker = '000100'
    code = '101110011'

    sim = generate_ccd_signal(
        code_bits=code,
        noise_sigma=500,
        vignette_strength=1,
        spread=0.5,
        marker=marker,
        seed=42
    )
    processed = process_signal(
        sim['adc_noisy'],
        roi_margin=3,
        min_pitch=3,
        max_pitch=3,
    )

    bits = cells_to_bits(processed.cell_values, threshold=0.5)
    bit_string = bits_to_string(bits)
    markers = find_marker(bits, marker=marker, max_errors=0)
    codes = extract_codes_after_markers(bits, marker=marker, code_len=9,
                                        max_errors=0)

    print('True bit sequence: ', sim['bit_seq'])
    print('Recovered cells:   ', bit_string)
    print('ROI:', processed.roi)
    print('Pitch:', processed.pitch)
    print('Phase:', processed.phase)
    print('Markers:', markers)
    print('Codes:', codes)

    plot_debug(processed, markers, codes)


if __name__ == '__main__':
    run_demo()
