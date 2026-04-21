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
    binary: NDArray
    pitch: float
    phase: float
    centers: NDArray
    cell_values: NDArray
    cell_bounds: list


# ---------------------------------------------------------------------------
# Огибающая
# ---------------------------------------------------------------------------

def estimate_envelope_morph(signal, structure_size=13, smooth_size=9):
    """
    Нижняя и верхняя огибающая через морфологическое открытие.

    structure_size подобран под 128-пиксельный кадр:
    при 3–4 пикселях на бит структура ~13 пикс охватывает ~3–4 бита,
    не срезая репер/маркер.
    """
    signal = np.asarray(signal, dtype=float)
    lower = grey_opening(signal, size=structure_size)
    upper = -grey_opening(-signal, size=structure_size)
    lower = uniform_filter1d(lower, size=smooth_size, mode="nearest")
    upper = uniform_filter1d(upper, size=smooth_size, mode="nearest")
    return lower, upper


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def detect_roi_from_envelope(signal, lower, upper, margin=5, min_width=24,
                              contrast_ratio=0.25):
    """
    Рабочая область по локальному контрасту upper − lower.
    """
    contrast = upper - lower
    cmax = np.max(contrast)
    if cmax <= 1e-9:
        return 0, len(signal)
    mask = contrast > contrast_ratio * cmax
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0, len(signal)
    left  = max(0, idx[0]  - margin)
    right = min(len(signal), idx[-1] + margin + 1)
    if right - left < min_width:
        return 0, len(signal)
    return left, right


# ---------------------------------------------------------------------------
# Нормировка
# ---------------------------------------------------------------------------

def normalize_signal(signal, lower, upper, left, right, eps=1e-6):
    """
    Нормировка по огибающим.
    Вне [left, right] значение равно 0.5 (нейтраль) — не ноль,
    чтобы не создавать ложных переходов на границе ROI.
    """
    signal = np.asarray(signal, dtype=float)
    denom = np.maximum(upper - lower, eps)
    norm = (signal - lower) / denom
    norm = np.clip(norm, 0.0, 1.0)
    out = np.full_like(norm, 0.5)   # нейтральное значение вне ROI
    out[left:right] = norm[left:right]
    return out


# ---------------------------------------------------------------------------
# Бинаризация (Bradley) — только внутри ROI
# ---------------------------------------------------------------------------

def bradley_threshold_1d_roi(signal, left, right, window_size=15, t=0.15):
    """
    Адаптивная бинаризация Брэдли только внутри [left, right].

    Вне ROI бинарный сигнал = 0 (не участвует в анализе).
    Порог считается по скользящему среднему только на участке ROI,
    поэтому нули вне окна не влияют на порог внутри.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    binary = np.zeros(n, dtype=np.uint8)

    if right <= left:
        return binary

    roi_sig = signal[left:right]
    # Скользящее среднее внутри ROI
    local_mean = uniform_filter1d(roi_sig, size=window_size, mode="nearest")
    threshold = local_mean * (1.0 - t)
    binary[left:right] = (roi_sig > threshold).astype(np.uint8)
    return binary


# ---------------------------------------------------------------------------
# Очистка от коротких серий
# ---------------------------------------------------------------------------

def remove_short_runs(binary, min_run_length=2):
    """
    Удаление коротких серий (шум на переходах).
    Работает только с ненулевыми участками.
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
            left_val  = b[start - 1] if start > 0 else value
            right_val = b[end]       if end   < n else value
            fill_val  = left_val if left_val == right_val else value
            b[start:end] = fill_val
        start = end
    return b


# ---------------------------------------------------------------------------
# Переходы
# ---------------------------------------------------------------------------

def find_transitions(binary, left, right):
    """
    Позиции переходов 0→1 и 1→0 внутри [left, right).
    """
    b = binary[left:right]
    d = np.diff(b.astype(int))
    return np.where(d != 0)[0] + left + 1


# ---------------------------------------------------------------------------
# Оценка шага (pitch) — устойчивая версия
# ---------------------------------------------------------------------------

def estimate_bit_pitch(transitions, min_pitch=2, max_pitch=20):
    """
    Оценка шага сетки битов.

    Алгоритм:
    1. Берём все интервалы между переходами.
    2. Отфильтровываем выбросы (< min_pitch или > max_pitch).
    3. 10-й перцентиль даёт оценку минимального интервала (≈ 1 бит).
    4. Для каждого интервала вычисляем ближайшее целое кратное к этому минимуму.
    5. Нормированные интервалы (dx / кратное) дают выборку оценок одного шага.
    6. Медиана этой выборки — итоговый pitch.

    Это устойчиво к пропущенным переходам (длинным сериям одного значения).
    """
    if len(transitions) < 2:
        raise ValueError("Недостаточно переходов для оценки шага")

    dx = np.diff(transitions.astype(float))
    dx = dx[(dx >= min_pitch) & (dx <= max_pitch)]

    if len(dx) == 0:
        raise ValueError("Нет переходов в допустимом диапазоне")

    # Оценка минимального шага (1 бит)
    pitch_1bit = float(np.percentile(dx, 10))

    # Кратные: 1, 2, 3, ...
    ratios = np.round(dx / pitch_1bit).astype(int)
    ratios = np.clip(ratios, 1, 8)

    normalized = dx / ratios          # оценки шага одного бита
    pitch = float(np.median(normalized))

    if pitch < min_pitch or pitch > max_pitch:
        raise ValueError(f"Оценённый шаг {pitch:.2f} вне допустимого диапазона")

    return pitch


# ---------------------------------------------------------------------------
# Оптимальная фаза сетки
# ---------------------------------------------------------------------------

def optimize_phase(left, transitions, pitch):
    """
    Фаза сетки (позиция первого центра ячейки).

    Переходы должны попадать на границы ячеек, т.е. на half-pitch от центра.
    Медиана остатков от деления на pitch → типичная граница → центр = граница + pitch/2.
    """
    transitions = np.asarray(transitions, dtype=float)
    if len(transitions) == 0:
        return left + pitch / 2.0
    remainders = transitions % pitch
    median_rem = float(np.median(remainders))
    center_offset = (median_rem + pitch / 2.0) % pitch
    # Смещаем так, чтобы первый центр был внутри ROI
    phase = left + (center_offset - left % pitch) % pitch
    return phase


# ---------------------------------------------------------------------------
# Сетка центров ячеек
# ---------------------------------------------------------------------------

def build_sampling_grid(left, right, pitch, phase):
    """
    Центры ячеек от phase с шагом pitch, ограниченные [left, right).
    """
    # Сдвигаем phase в первую позицию >= left
    if phase < left:
        n_skip = int(np.ceil((left - phase) / pitch))
        phase = phase + n_skip * pitch
    centers = []
    x = phase
    while x < right:
        centers.append(x)
        x += pitch
    return np.array(centers, dtype=float)


# ---------------------------------------------------------------------------
# Сэмплирование битов — стиль find_datablock из main.c
# ---------------------------------------------------------------------------

def sample_bits(signal, lower, upper, centers, pitch, left, right, eps=1e-6):
    """
    Извлечение значения каждого бита методом, близким к main.c find_datablock.

    Для каждого центра ячейки:
    - Берём среднее сигнала в окне [center - pitch/2, center + pitch/2].
    - Нормируем: 0 = тёмный (нижняя огибающая), 1 = светлый (верхняя огибающая).
      Граница — 0.5.

    Используем RAW сигнал + локальные огибающие (а не глобально нормированный),
    что соответствует логике main.c: pix[i][1] = t_data - min_sdata.

    Возвращает:
        cell_values  — float [0, 1] для каждого центра (вне ROI = NaN)
        cell_bounds  — список пар (a, b) границ окна
    """
    signal = np.asarray(signal, dtype=float)
    half = pitch / 2.0
    values = []
    bounds = []

    # Локальные min/max внутри ROI (аналог max_sdata / min_sdata)
    roi_sig = signal[left:right]
    roi_lower = lower[left:right]
    roi_upper = upper[left:right]
    local_min = float(np.min(roi_sig))
    local_max = float(np.max(roi_sig))
    denom_global = max(local_max - local_min, eps)

    for c in centers:
        a = int(np.floor(c - half))
        b = int(np.ceil(c + half))
        a_clamp = max(left,  a)
        b_clamp = min(right, b)

        if b_clamp <= a_clamp:
            # Центр вне ROI — ставим NaN
            values.append(float("nan"))
            bounds.append((a, b))
            continue

        raw_window  = signal[a_clamp:b_clamp]
        lo_window   = lower[a_clamp:b_clamp]
        hi_window   = upper[a_clamp:b_clamp]

        # Усредняем raw и огибающие в окне
        raw_mean = float(np.mean(raw_window))
        lo_mean  = float(np.mean(lo_window))
        hi_mean  = float(np.mean(hi_window))

        denom = max(hi_mean - lo_mean, eps)
        val = (raw_mean - lo_mean) / denom
        val = float(np.clip(val, 0.0, 1.0))
        values.append(val)
        bounds.append((a_clamp, b_clamp))

    return np.array(values, dtype=float), bounds


# ---------------------------------------------------------------------------
# Главная функция pipeline
# ---------------------------------------------------------------------------

def process_signal(
    adc_signal,
    envelope_size=13,
    envelope_smooth=9,
    roi_margin=5,
    roi_min_width=24,
    roi_contrast_ratio=0.25,
    bradley_window=15,
    bradley_t=0.15,
    cleanup_min_run=2,
    min_pitch=2,
    max_pitch=20,
):
    """
    Полный pipeline обработки сигнала ПЗС-линейки.

    Шаги:
    1. Морфологическая огибающая (lower / upper).
    2. Определение ROI по контрасту огибающих.
    3. Нормировка: внутри ROI — по огибающим, вне — 0.5 (нейтраль).
    4. Адаптивная бинаризация Брэдли ТОЛЬКО внутри ROI.
    5. Удаление коротких серий (шум).
    6. Поиск переходов внутри ROI.
    7. Устойчивая оценка шага (pitch) через медиану кратных.
    8. Оптимизация фазы сетки.
    9. Построение сетки центров.
    10. Сэмплирование cell_values (стиль find_datablock).
    """
    adc_signal = np.asarray(adc_signal, dtype=float)
    n = len(adc_signal)

    # --- 1. Огибающая ---
    lower, upper = estimate_envelope_morph(
        adc_signal,
        structure_size=envelope_size,
        smooth_size=envelope_smooth,
    )

    # --- 2. ROI ---
    left, right = detect_roi_from_envelope(
        adc_signal, lower, upper,
        margin=roi_margin,
        min_width=roi_min_width,
        contrast_ratio=roi_contrast_ratio,
    )

    # --- 3. Нормировка ---
    normalized = normalize_signal(adc_signal, lower, upper, left, right)

    # --- 4. Bradley ТОЛЬКО внутри ROI ---
    binary = bradley_threshold_1d_roi(
        normalized, left, right,
        window_size=bradley_window,
        t=bradley_t,
    )

    # --- 5. Очистка ---
    binary = remove_short_runs(binary, min_run_length=cleanup_min_run)
    # Гарантируем, что вне ROI — нули
    binary[:left]  = 0
    binary[right:] = 0

    # --- 6. Переходы ---
    transitions = find_transitions(binary, left, right)

    # --- 7. Pitch ---
    pitch = estimate_bit_pitch(transitions, min_pitch=min_pitch, max_pitch=max_pitch)

    # --- 8. Фаза ---
    phase = optimize_phase(left, transitions, pitch)

    # --- 9. Сетка ---
    centers = build_sampling_grid(left, right, pitch, phase)

    # --- 10. Сэмплирование ---
    cell_values, cell_bounds = sample_bits(
        adc_signal, lower, upper, centers, pitch, left, right
    )

    return SignalProcessingResult(
        raw=adc_signal,
        lower_envelope=lower,
        upper_envelope=upper,
        roi=(left, right),
        normalized=normalized,
        binary=binary,
        pitch=pitch,
        phase=phase,
        centers=centers,
        cell_values=cell_values,
        cell_bounds=cell_bounds,
    )
