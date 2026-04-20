import matplotlib.pyplot as plt
import numpy as np

from simulation import generate_ccd_signal
from signal_processing import process_signal
from marker_code import cells_to_bits, bits_to_string, find_marker, extract_codes_after_markers


def plot_debug(processed):
    raw = processed.raw
    lower = processed.lower_envelope
    upper = processed.upper_envelope
    norm = processed.normalized
    thr = processed.local_threshold
    binary = processed.binary
    left, right = processed.roi
    centers = processed.centers

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(raw, color='black', lw=1.2, label='raw')
    ax[0].plot(lower, '--', lw=1.0, label='lower envelope')
    ax[0].plot(upper, '--', lw=1.0, label='upper envelope')
    ax[0].axvline(left, color='tab:red', ls=':')
    ax[0].axvline(right, color='tab:red', ls=':')
    ax[0].legend()
    ax[0].set_ylabel('ADC')

    ax[1].plot(norm, color='tab:blue', lw=1.2, label='normalized')
    ax[1].plot(thr, color='tab:orange', lw=1.0, label='Bradley threshold')
    edges = (centers[1:] + centers[:-1]) / 2
    for e in edges:
        ax[1].axvline(e, color='gray', alpha=0.2)
    ax[1].legend()
    ax[1].set_ylabel('Norm')

    ax[2].step(range(len(binary)), binary, where='mid', color='black', label='binary')
    for e in edges:
        ax[2].axvline(e, color='gray', alpha=0.2)
    ax[2].axvline(left, color='tab:red', ls=':')
    ax[2].axvline(right, color='tab:red', ls=':')
    ax[2].set_ylabel('Bit')
    ax[2].set_xlabel('Pixel')
    ax[2].legend()

    plt.tight_layout()
    plt.show()


def run_demo():
    sim = generate_ccd_signal(code_bits='100111101', marker='000100', seed=42)
    processed = process_signal(sim['adc_noisy'], min_pitch=2, max_pitch=10)

    bits = cells_to_bits(processed.cell_values, threshold=0.5)
    bits = np.pad(
       bits,
        (2, 2)
    )
    bit_string = bits_to_string(bits)
    markers = find_marker(bits, marker='000100', max_errors=0)
    codes = extract_codes_after_markers(bits, marker='000100', code_len=9, max_errors=0)

    print('True bit sequence: ', sim['bit_seq'])
    print('Recovered cells:   ', bit_string)
    print('ROI:', processed.roi)
    print('Pitch:', processed.pitch)
    print('Phase:', processed.phase)
    print('Centers:', processed.centers)
    print('Markers:', markers)
    print('Codes:', codes)

    plot_debug(processed)


if __name__ == '__main__':
    run_demo()
