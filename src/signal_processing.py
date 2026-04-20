from dataclasses import dataclass
import numpy as np
from scipy.ndimage import grey_opening, uniform_filter1d
from numpy.typing import NDArray

@dataclass
class SignalProcessingResult:
    raw: NDArray
    lower_envelope: NDArray
    upper_envelope: NDArray
    roi: tuple
    normalized: NDArray
    local_threshold: NDArray
    binary: NDArray
    transitions: NDArray
    pitch: float
    phase: float
    centers: NDArray[float]
    cell_values: NDArray
    cell_bounds: list



def estimate_envelope_morph(signal, structure_size=21, smooth_size=15):
    """
    Оценка нижней и верхней огибающей используя морфологические операции
    Операция открытия на исходном сигнале дает нижний уровень
    На дополнительном сигнале -- верхний уровень
    """

    signal = np.asarray(signal, dtype=float)

    lower = grey_opening(signal, size=structure_size)
    upper = -grey_opening(-signal, size=structure_size)

    lower = uniform_filter1d(lower, size=smooth_size, mode="nearest")
    upper = uniform_filter1d(upper, size=smooth_size, mode="nearest")

    return lower, upper



def detect_roi_from_envelope(signal, lower, upper, margin=3, min_width=24, contrast_ratio=0.5):
    """
    Поиск рабочей области по локальному контрасту upper-lower.
    """

    contrast = upper - lower
    cmax = np.max(contrast)

    if cmax <= 1e-9:
        return 0, len(signal)

    mask = contrast > contrast_ratio * cmax
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return 0, len(signal)

    left = max(0, idx[0] - margin)
    right = min(len(signal), idx[-1] + margin + 1)

    if right - left < min_width:
        return 0, len(signal)

    return left, right



def normalize_signal(signal, lower, upper, left=None, right=None, eps=1e-6):
    """
    Нормировка по морфологическим огибающим.
    """

    signal = np.asarray(signal, dtype=float)
    denom = np.maximum(upper - lower, eps)
    norm = (signal - lower) / denom
    norm = np.clip(norm, 0.0, 1.0)

    if left is not None and right is not None:
        out = np.zeros_like(norm)
        out[left:right] = norm[left:right]
        return out

    return norm



def bradley_threshold_1d(signal, window_size=15, t=0.15):
    """
    1D бинаризация методом Брэдли.
    """

    signal = np.asarray(signal, dtype=float)
    local_mean = uniform_filter1d(signal, size=window_size, mode="nearest")
    threshold = local_mean * (1.0 - t)
    binary = (signal > threshold).astype(np.uint8)
    return binary, threshold



def remove_short_runs(binary, min_run_length=2):
    """
    Удаление очень коротких серий 01 10, вызванных шумом.
    """

    b = np.asarray(binary, dtype=np.uint8).copy()
    n = len(b)
    start = 0

    while start < n:
        value = b[start]
        end = start + 1
        while end < n and b[end] == value:
            end += 1

        run_len = end - start
        if run_len < min_run_length:
            left_val = b[start - 1] if start > 0 else value
            right_val = b[end] if end < n else value
            fill_val = left_val if left_val == right_val else value
            b[start:end] = fill_val

        start = end

    return b



def find_transitions(binary, left=None, right=None):
    """
    Поиск всех переходов 0-1 и 1-0.
    """

    if left is None:
        left = 0
    if right is None:
        right = len(binary)

    b = binary[left:right]
    d = np.diff(b.astype(int))
    return np.where(d != 0)[0] + left + 1



def estimate_bit_pitch(transitions, min_pitch=2, max_pitch=20):
    """
    Поиск длины бита как минимальная длина переходов
    (предварительно требуется очистка от мусорных быстрых переходов)
    """

    if len(transitions) < 2:
        raise ValueError("Недостаточно переходов для оценки шага сетки")

    dx = np.diff(transitions)
    dx = dx[(dx >= min_pitch) & (dx <= max_pitch)]

    if len(dx) == 0:
        raise ValueError("Не удалось оценить шаг сетки")

    return float(np.min(dx))



def build_sampling_grid(left, right, pitch, phase=None):
    """
    Генерация сетки между left и right с шагом pitch и началом в phase
    """

    if phase is None:
        phase = left + pitch / 2.0

    centers = []
    x = phase
    while x < right:
        centers.append(x)
        x += pitch

    return np.array(centers, dtype=float)



def integrate_cells(signal, centers, pitch, left=0, right=None):
    """
    Объединение пикселов в пределах узлов сетки в единый средний бит
    """

    if right is None:
        right = len(signal)

    signal = np.asarray(signal, dtype=float)
    half = pitch / 2.0
    values = []
    bounds = []

    for c in centers:
        a = max(left, int(np.floor(c - half)))
        b = min(right, int(np.ceil(c + half)))
        if b <= a:
            continue
        values.append(np.mean(signal[a:b]))
        bounds.append((a, b))

    return np.array(values, dtype=float), bounds


def optimize_phase(left, transitions, pitch):
    """
    Оптимальная фаза для центров ячеек.

    1. remainders --- положение переходов в ячейке
    2. median_remainder --- типичное положение границы ячейки
    3. center_of_cell = median_remainder + pitch/2 --- центр ячейки
    4. phase = left + (center_of_cell % pitch) --- первый центр в ROI
    """

    transitions = np.asarray(transitions, dtype=float)
    remainders = transitions % pitch
    median_remainder = np.median(remainders)

    center_offset = (median_remainder + pitch / 2.0) % pitch
    phase = left + center_offset

    return phase



def process_signal(
    adc_signal,
    envelope_size=21,
    envelope_smooth=15,
    roi_margin=3,
    roi_min_width=24,
    roi_contrast_ratio=0.25,
    bradley_window=15,
    bradley_t=0.15,
    cleanup_min_run=2,
    min_pitch=2,
    max_pitch=20,
):
    """
    Полный пайплайн обработки сигнала до уровня битовых ячеек.
    """

    adc_signal = np.asarray(adc_signal, dtype=float)

    lower, upper = estimate_envelope_morph(
        adc_signal,
        structure_size=envelope_size,
        smooth_size=envelope_smooth,
    )

    left, right = detect_roi_from_envelope(
        adc_signal,
        lower,
        upper,
        margin=roi_margin,
        min_width=roi_min_width,
        contrast_ratio=roi_contrast_ratio,
    )

    normalized = normalize_signal(adc_signal, lower, upper, left=left, right=right)

    binary, local_threshold = bradley_threshold_1d(
        normalized,
        window_size=bradley_window,
        t=bradley_t,
    )

    binary[:left] = 0
    binary[right:] = 0
    binary = remove_short_runs(binary, min_run_length=cleanup_min_run)

    transitions = find_transitions(binary, left=left, right=right)
    pitch = estimate_bit_pitch(transitions, min_pitch=min_pitch, max_pitch=max_pitch)
    phase = optimize_phase(left, transitions, pitch)
    centers = build_sampling_grid(left, right, pitch, phase=phase)
    cell_values, cell_bounds = integrate_cells(normalized, centers, pitch, left=left, right=right)

    return SignalProcessingResult(
        raw=adc_signal,
        lower_envelope=lower,
        upper_envelope=upper,
        roi=(left, right),
        normalized=normalized,
        local_threshold=local_threshold,
        binary=binary,
        transitions=transitions,
        pitch=pitch,
        phase=phase,
        centers=centers,
        cell_values=cell_values,
        cell_bounds=cell_bounds,
    )
