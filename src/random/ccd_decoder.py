"""
ccd_decoder.py — модуль субпиксельного восстановления битовой последовательности

Pipeline обработки:
1. Компенсация виньетирования (скользящий максимум)
2. КИХ-фильтр 8-го порядка Блэ-Риу → производная
3. Поиск пиков |производной| → грубые позиции фронтов
4. Уточнение фронтов: параболическая интерполяция ИЛИ нулевой переход производной (Blais-Rioux)
5. МНК-подгонка тактовой сетки (оценка T и x0)
6. Сэмплирование центров бит + адаптивный порог → декодированные биты

Дополнительно вычисляются:
- метрики точности позиций фронтов (если переданы истинные фронты)
- точность декодирования бит (если передана истинная последовательность)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
from scipy.signal import find_peaks


@dataclass
class DecoderConfig:
    bit_width_hint_px: Optional[float] = None
    """Приблизительная ширина бита в пикселях (None — оценивается автоматически)."""
    vignette_window_factor: float = 3.0
    """Ширина окна скользящего максимума = round(bit_width * factor)."""
    fir_order: Literal[4, 8] = 8
    """Порядок КИХ-фильтра Блэ-Риу (4 или 8)."""
    edge_threshold_rel: float = 0.25
    """Относительный порог амплитуды пиков производной (от максимума)."""
    subpixel_method: Literal["parabola", "zero_crossing"] = "zero_crossing"
    """
    Метод субпиксельного уточнения позиций фронтов:
    - 'parabola'       — параболическая интерполяция вершины |d|
    - 'zero_crossing'  — линейная интерполяция нулевого перехода d (метод Блэ-Риу)
    """
    lsq_clock_recovery: bool = True
    """Использовать МНК-подгонку тактовой сетки по всем найденным фронтам."""
    adaptive_threshold: bool = True
    """
    True  — адаптивный порог: среднее между глобальным min и max нормированного сигнала.
    False — фиксированный порог 0.5.
    """


@dataclass
class DecoderResult:
    decoded_bits: np.ndarray
    """Восстановленная битовая последовательность."""
    bit_centers_px: np.ndarray
    """Субпиксельные позиции центров бит в пикселях ПЗС."""
    edge_positions_px: np.ndarray
    """Субпиксельные позиции найденных фронтов в пикселях ПЗС."""
    clock_period_px: float
    """Оценённый период тактовой сетки T в пикселях ПЗС."""
    clock_phase_px: float
    """Оценённая начальная фаза тактовой сетки x0 (позиция нулевого фронта)."""
    signal_normalized: np.ndarray
    """Нормированный сигнал после компенсации виньетирования."""
    derivative: np.ndarray
    """Сигнал производной (выход КИХ-фильтра Блэ-Риу)."""
    threshold: float
    """Использованный порог бинаризации."""
    # ---- метрики качества (заполняются если переданы истинные данные) ----
    accuracy: Optional[float] = None
    """Точность декодирования бит [0..1]."""
    edge_rmse_px: Optional[float] = None
    """СКО ошибок позиций фронтов в пикселях."""
    edge_bias_px: Optional[float] = None
    """Систематическое смещение позиций фронтов в пикселях."""
    clock_period_error_px: Optional[float] = None
    """Ошибка оценки периода тактовой сетки."""
    n_edges_found: int = 0
    n_edges_true: int = 0


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _blais_rioux_fir(order: Literal[4, 8]) -> np.ndarray:
    """Возвращает нормированный КИХ-фильтр Блэ-Риу заданного порядка."""
    if order == 4:
        h = np.array([1., 1., -1., -1.], dtype=float)
    else:
        h = np.array([1., 1., 1., 1., -1., -1., -1., -1.], dtype=float)
    h /= h[h > 0].sum()
    return h


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Скользящий максимум с дублированием краёв."""
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    result = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        result[i] = padded[i: i + window].max()
    return result


def _parabola_subpixel(arr: np.ndarray, idx: int) -> float:
    """Субпиксельная вершина параболы по трём точкам arr[idx-1..idx+1]."""
    if idx < 1 or idx >= len(arr) - 1:
        return float(idx)
    y0, y1, y2 = arr[idx - 1], arr[idx], arr[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(idx)
    delta = np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)
    return idx + delta


def _zero_crossing_subpixel(d: np.ndarray, idx: int) -> float:
    """
    Субпиксельный нулевой переход производной вблизи пика |d|.
    Находим переход знака d в окрестности idx, линейная интерполяция (метод Блэ-Риу).
    """
    # Ищем пересечение нуля в окне ±2 от пика |d|
    search = range(max(0, idx - 2), min(len(d) - 1, idx + 3))
    for i in search:
        a, b = d[i], d[i + 1] if i + 1 < len(d) else d[i]
        if a * b < 0:
            # Линейная интерполяция: x = i + A/(A-B)
            return i + a / (a - b)
    return float(idx)


# ---------------------------------------------------------------------------
# Основная функция декодирования
# ---------------------------------------------------------------------------

def decode_ccd(
    adc_signal: np.ndarray,
    config: DecoderConfig = None,
    true_edges: Optional[np.ndarray] = None,
    true_bits: Optional[np.ndarray] = None,
    true_bit_width: Optional[float] = None,
) -> DecoderResult:
    """
    Восстанавливает битовую последовательность из сигнала ПЗС-линейки.

    Parameters
    ----------
    adc_signal : np.ndarray
        Одномерный массив значений АЦП (целые или вещественные).
    config : DecoderConfig, optional
        Параметры алгоритма. None → значения по умолчанию.
    true_edges : np.ndarray, optional
        Истинные позиции фронтов (для вычисления метрик точности).
    true_bits : np.ndarray, optional
        Истинная битовая последовательность (для вычисления accuracy).
    true_bit_width : float, optional
        Истинная ширина бита (для вычисления ошибки оценки T).

    Returns
    -------
    DecoderResult
    """
    if config is None:
        config = DecoderConfig()

    sig = adc_signal.astype(float)
    n = len(sig)
    px = np.arange(n, dtype=float)

    # Оценка ширины бита для настройки параметров
    bw = config.bit_width_hint_px if config.bit_width_hint_px is not None else max(4.0, n / 32.0)

    # ------------------------------------------------------------------
    # 1. Компенсация виньетирования (скользящий максимум)
    # ------------------------------------------------------------------
    win_v = max(3, round(bw * config.vignette_window_factor))
    rolling_max = _rolling_max(sig, win_v)
    rolling_max = np.maximum(rolling_max, 0.05 * sig.max())
    sig_norm = sig / rolling_max

    # ------------------------------------------------------------------
    # 2. КИХ-фильтр Блэ-Риу → производная
    # ------------------------------------------------------------------
    fir = _blais_rioux_fir(config.fir_order)
    derivative = np.convolve(sig_norm, fir, mode="same")

    # ------------------------------------------------------------------
    # 3. Поиск пиков |производной|
    # ------------------------------------------------------------------
    abs_d = np.abs(derivative)
    thresh_d = config.edge_threshold_rel * derivative.max()
    min_dist = max(2, int(bw * 0.3))
    peak_idx, _ = find_peaks(derivative, height=thresh_d, distance=min_dist)

    # ------------------------------------------------------------------
    # 4. Субпиксельное уточнение позиций фронтов
    # ------------------------------------------------------------------
    edge_positions = []
    for pi in peak_idx:
        if config.subpixel_method == "zero_crossing":
            ep = _zero_crossing_subpixel(derivative, int(pi))
        else:
            ep = _parabola_subpixel(derivative, int(pi))
        edge_positions.append(ep)

    edge_positions = np.sort(np.array(edge_positions, dtype=float))

    # Отсечение по краям (шум на границах массива)
    margin = max(1, int(bw * 0.5))
    edge_positions = edge_positions[(edge_positions > margin) & (edge_positions < n - margin)]

    # ------------------------------------------------------------------
    # 5. Оценка тактовой сетки (T, x0) через МНК
    # ------------------------------------------------------------------
    T_fit, x0_fit = _estimate_clock(edge_positions, bw, n, margin, config)

    # Построение центров бит
    k_start = int(np.floor((margin - x0_fit) / T_fit)) - 1
    k_end = int(np.ceil((n - margin - x0_fit) / T_fit)) + 1
    bit_centers = []
    for k in range(k_start, k_end + 1):
        c = x0_fit + T_fit * (k + 0.5)
        if margin <= c <= n - margin:
            bit_centers.append(c)
    bit_centers = np.array(bit_centers)

    # ------------------------------------------------------------------
    # 6. Бинаризация: сэмплирование + порог
    # ------------------------------------------------------------------
    if len(bit_centers) > 0:
        sig_at_centers = np.interp(bit_centers, px, sig_norm)
        if config.adaptive_threshold:
            threshold = (sig_norm.min() + sig_norm.max()) / 2.0
        else:
            threshold = 0.5
        decoded = (sig_at_centers > threshold).astype(np.uint8)
    else:
        sig_at_centers = np.array([])
        threshold = 0.5
        decoded = np.array([], dtype=np.uint8)

    # ------------------------------------------------------------------
    # 7. Метрики качества (опционально)
    # ------------------------------------------------------------------
    accuracy = None
    edge_rmse = None
    edge_bias = None
    clock_err = None
    inner_true = None

    if true_bits is not None and len(decoded) > 0:
        accuracy = _best_alignment_accuracy(true_bits, decoded)

    if true_edges is not None and len(edge_positions) > 0:
        inner_true = true_edges[1:-1] if len(true_edges) > 2 else true_edges
        errs = _match_edge_errors(edge_positions, inner_true, T_fit)
        if len(errs) > 0:
            edge_rmse = float(np.sqrt(np.mean(errs ** 2)))
            edge_bias = float(np.mean(errs))

    if true_bit_width is not None:
        clock_err = float(T_fit - true_bit_width)

    return DecoderResult(
        decoded_bits=decoded,
        bit_centers_px=bit_centers,
        edge_positions_px=edge_positions,
        clock_period_px=T_fit,
        clock_phase_px=x0_fit,
        signal_normalized=sig_norm,
        derivative=derivative,
        threshold=threshold,
        accuracy=accuracy,
        edge_rmse_px=edge_rmse,
        edge_bias_px=edge_bias,
        clock_period_error_px=clock_err,
        n_edges_found=len(edge_positions),
        n_edges_true=len(inner_true) if inner_true is not None else 0,
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции: тактовая сетка и метрики
# ---------------------------------------------------------------------------

def _estimate_clock(
    edge_positions: np.ndarray,
    bw_hint: float,
    n_pixels: int,
    margin: int,
    config: DecoderConfig,
) -> tuple[float, float]:
    """
    Оценивает T (тактовый период) и x0 (начальная фаза) тактовой сетки.

    Если задан bit_width_hint_px — итерация стартует с него, что даёт более
    точный результат при большом числе одинаковых подряд идущих бит (нет фронтов
    → большие зазоры искажают медиану).
    """
    if len(edge_positions) < 2:
        T = bw_hint
        x0 = edge_positions[0] if len(edge_positions) == 1 else margin
        return T, x0

    gaps = np.diff(edge_positions)

    # Стартовая оценка T: hint имеет приоритет перед медианой зазоров,
    # потому что при длинных сериях одинаковых бит зазоры кратны T,
    # и медиана без hint завышает оценку.
    if config.bit_width_hint_px is not None:
        T_init = config.bit_width_hint_px
    else:
        T_med = np.median(gaps)
        multiples = np.maximum(np.round(gaps / T_med).astype(int), 1)
        T_init = np.sum(gaps) / np.sum(multiples)

    # Итеративное уточнение T + МНК-подгонка фазы
    T_fit = T_init
    for _ in range(5):
        mults = np.maximum(np.round(gaps / T_fit).astype(int), 1)
        T_new = np.sum(gaps) / np.sum(mults)
        if abs(T_new - T_fit) < 1e-6:
            T_fit = T_new
            break
        T_fit = T_new

    # МНК: edge_i = x0 + T * cum_idx_i  →  оценка x0 и финальное T
    mults = np.maximum(np.round(gaps / T_fit).astype(int), 1)
    cum_idx = np.concatenate([[0], np.cumsum(mults)]).astype(float)
    A = np.column_stack([np.ones(len(edge_positions)), cum_idx])
    params, *_ = np.linalg.lstsq(A, edge_positions, rcond=None)
    x0 = params[0]
    T_fit = max(params[1], 1.0)

    return T_fit, x0


def _match_edge_errors(
    found: np.ndarray, true: np.ndarray, T: float, match_tol: float = 0.6
) -> np.ndarray:
    """Сопоставляет найденные фронты с ближайшими истинными и возвращает ошибки."""
    errors = []
    for ep in found:
        dists = np.abs(true - ep)
        i_min = int(np.argmin(dists))
        if dists[i_min] < T * match_tol:
            errors.append(ep - true[i_min])
    return np.array(errors)


def _best_alignment_accuracy(true: np.ndarray, decoded: np.ndarray) -> float:
    """Точность декодирования с учётом возможного сдвига последовательности."""
    best_acc = 0.0
    n = len(true)
    m = len(decoded)
    for offset in range(-(m - 1), n):
        if offset >= 0:
            t_slice = true[offset: offset + m]
            d_slice = decoded[: len(t_slice)]
        else:
            d_slice = decoded[-offset:]
            t_slice = true[: len(d_slice)]
        length = min(len(t_slice), len(d_slice))
        if length < 1:
            continue
        acc = float(np.mean(t_slice[:length] == d_slice[:length]))
        if acc > best_acc:
            best_acc = acc
    return best_acc
