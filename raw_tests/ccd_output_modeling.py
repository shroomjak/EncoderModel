import numpy as np
import matplotlib.pyplot as plt

# ----- Параметры ПЗС -----
L_ARRAY = 128          # число пикселей
ADC_MAX = 4095         # 12 бит

# ----- Физико-геометрические параметры (подстрой) -----
BIT_PIXELS = 3         # базовое число субпикселов на бит до свёртки
SPREAD = 0.5           # сигма гауссовой PSF в субпикселах (сглаживание, ~ширина щели)
VIGNETTE_STRENGTH = 0.5  # 0..1, чем больше, тем сильнее спад к краям
NOISE_SIGMA = 150      # СКО шума (кванты АЦП)

# ----- Заданный код -----
code_bits = "111111"

# ----- 1. Шаблон: шум + репер + код + репер + код + репер + шум -----
marker = "000100"       # пример репера
noise_bits_left = "0" * 10
noise_bits_right = "0" * 10

bit_seq = (
    noise_bits_left
    + marker
    + code_bits
    + marker
    + code_bits
    + marker
    + noise_bits_right
)

# ----- 2. Развёртка битов в субпиксели (идеальный прямоугольный сигнал) -----
bright_level = 1.0     # "1" — светлый
dark_level = 0.0       # "0" — тёмный

signal_len = len(bit_seq) * BIT_PIXELS
base = np.zeros(signal_len)

for i, b in enumerate(bit_seq):
    level = bright_level if b == "1" else dark_level
    start = i * BIT_PIXELS
    end = start + BIT_PIXELS
    base[start:end] = level

# ----- 2b. Сглаживание: свёртка с гауссом (имитация оптики / неравномерной засветки) -----
if SPREAD > 0:
    radius = int(BIT_PIXELS * SPREAD)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / SPREAD) ** 2)
    kernel /= kernel.sum()
    blurred = np.convolve(base, kernel, mode="same")
else:
    blurred = base.copy()

# ----- 3. Ресэмплинг в 128 реальных пикселей ПЗС -----
# интерпретируем blurred как непрерывный профиль интенсивности,
# интегрируем по равным отрезкам -> значение пикселя
edges = np.linspace(0, len(blurred), L_ARRAY + 1)
pixel_vals = np.zeros(L_ARRAY)

for i in range(L_ARRAY):
    s = int(np.floor(edges[i]))
    e = int(np.floor(edges[i + 1]))
    if e <= s:
        e = s + 1
    segment = blurred[s:e]
    pixel_vals[i] = segment.mean() if len(segment) > 0 else 0.0

# нормировка в [0, 1]
pixel_vals -= pixel_vals.min()
if pixel_vals.max() > 0:
    pixel_vals /= pixel_vals.max()

# ----- 3b. Виньетирование -----
# плавный спад к краям, центр ~1
idx = np.linspace(-1, 1, L_ARRAY)
# форма можно менять; сейчас плавная косинусная
vignette = 1.0 - VIGNETTE_STRENGTH * (1 - np.cos(np.pi * idx / 2))
pixel_vals *= vignette

# ----- 4. Перевод в уровни АЦП и добавление шума -----
adc_vals = pixel_vals * ADC_MAX
noise = np.random.normal(0, NOISE_SIGMA, size=L_ARRAY)
adc_noisy = adc_vals + noise
adc_noisy = np.clip(np.round(adc_noisy), 0, ADC_MAX).astype(int)

# ----- Визуализация -----
plt.figure(figsize=(10, 3))

# "Изображение" — одна строка из 128 пикселей
plt.subplot(2, 1, 1)
plt.imshow(adc_noisy[np.newaxis, :], cmap="gray", aspect="auto",
           vmin=0, vmax=ADC_MAX)
plt.yticks([])
plt.title("Смоделированная линия ПЗС")

# Профиль яркости
plt.subplot(2, 1, 2)
plt.plot(adc_noisy, "-k")
plt.ylim(0, ADC_MAX)
plt.xlabel("Номер пикселя")
plt.ylabel("Значение АЦП")
plt.tight_layout()
plt.show()

# Для отладки можно посмотреть первые несколько значений
print("First 16 pixels:", adc_noisy[:16])