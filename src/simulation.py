import numpy as np
import matplotlib.pyplot as plt

ADC_MAX = 4095


def generate_ccd_signal(
    code_bits="111111",
    marker="000100",
    l_array=128,
    bit_pixels=3,
    spread=0.5,
    vignette_strength=0.75,
    noise_sigma=150,
    bright_level=1.0,
    dark_level=0.0,
    seed=None,
):
    """
    Генерация 1D сигнала ПЗС-линейки для абсолютного энкодера.

    Структура сигнала:
        noise_left + marker + code + marker + code + marker + noise_right

    Возвращает словарь с промежуточными стадиями моделирования.
    """
    rng = np.random.default_rng(seed)

    bit_seq = (
        marker
        + code_bits
        + marker
        + code_bits
        + marker
    )

    signal_len = len(bit_seq) * bit_pixels
    base = np.zeros(signal_len, dtype=float)

    for i, b in enumerate(bit_seq):
        level = bright_level if b == "1" else dark_level
        start = i * bit_pixels
        end = start + bit_pixels
        base[start:end] = level

    if spread > 0:
        radius = max(1, int(np.ceil(bit_pixels * spread * 3)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / spread) ** 2)
        kernel /= kernel.sum()
        blurred = np.convolve(base, kernel, mode="same")
    else:
        blurred = base.copy()

    remain = l_array - len(blurred)
    pixel_vals = blurred.copy()

    if remain > 0:
        pixel_vals = np.pad(
            pixel_vals,
            [(remain // 2, remain - remain // 2)],
            mode="constant"
        )
    elif remain < 0:
        pixel_vals = pixel_vals[:l_array]

    idx = np.linspace(-1, 1, l_array)
    vignette = 1.0 - vignette_strength * (1 - np.cos(np.pi * idx / 2))
    pixel_vals *= vignette

    adc_vals = pixel_vals * ADC_MAX
    noise = rng.normal(0, noise_sigma, size=l_array)
    adc_noisy = adc_vals + noise
    adc_noisy = np.clip(np.round(adc_noisy), 0, ADC_MAX).astype(int)

    return {
        "bit_seq": bit_seq,
        "pixel_vals": pixel_vals,
        "adc_vals": adc_vals,
        "adc_noisy": adc_noisy,
        "params": {
            "l_array": l_array,
            "bit_pixels": bit_pixels,
            "spread": spread,
            "vignette_strength": vignette_strength,
            "noise_sigma": noise_sigma,
            "marker": marker,
            "code_bits": code_bits,
        },
    }


if __name__ == "__main__":
    meta_signal = generate_ccd_signal()
    # ----- Визуализация -----
    plt.figure(figsize=(10, 3))

    # "Изображение" — одна строка из 128 пикселей
    plt.subplot(2, 1, 1)
    plt.imshow(meta_signal["adc_noisy"][np.newaxis, :], cmap="gray", aspect="auto",
               vmin=0, vmax=ADC_MAX)
    plt.yticks([])
    plt.title("Смоделированная линия ПЗС")

    # Профиль яркости
    plt.subplot(2, 1, 2)
    plt.plot(meta_signal["adc_noisy"], "-k")
    plt.ylim(0, ADC_MAX)
    plt.xlabel("Номер пикселя")
    plt.ylabel("Значение АЦП")
    plt.tight_layout()
    plt.show()