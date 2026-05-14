from dataclasses import dataclass
import numpy as np


@dataclass
class MarkerMatch:
    start_index: int
    errors: int
    matched_bits: str

    def __repr__(self):
        return f'Marker(start={self.start_index}, err={self.errors})'


@dataclass
class ExtractedCode:
    marker_start: int
    marker_errors: int
    code_start: int
    code_bits: np.ndarray
    code_str: str

    def __repr__(self):
        return f'Code(after={self.marker_start}, start={self.code_start}, bits={self.code_str})'


def cells_to_bits(cell_values, threshold=0.5):
    """
    Перевод float cell_values в бинарный массив.

    Ячейки вне ROI могут содержать NaN (sample_bits возвращает NaN
    для центров, не попавших в ROI). Такие ячейки трактуются как 0,
    чтобы не ломать поиск маркера на краях.
    """
    v = np.asarray(cell_values, dtype=float)
    bits = np.zeros(len(v), dtype=np.uint8)
    valid = ~np.isnan(v)
    bits[valid] = (v[valid] > threshold).astype(np.uint8)
    return bits


def bits_to_string(bits):
    return ''.join(str(int(b)) for b in bits)


def hamming_distance_str(a, b):
    return sum(x != y for x, y in zip(a, b))


def find_marker(bits, marker, max_errors=1):
    """
    Поиск маркера по расстоянию Хэмминга.
    """

    bit_str = bits_to_string(bits)
    m = len(marker)
    found = []

    for i in range(len(bit_str) - m + 1):
        window = bit_str[i:i + m]
        err = hamming_distance_str(window, marker)
        if err <= max_errors:
            found.append(MarkerMatch(i, err, window))

    return found



def extract_codes_after_markers(bits, marker, code_len=6, max_errors=1):
    """
    Извлечение кодовых слов сразу после найденных маркеров.
    """

    bits = np.asarray(bits, dtype=np.uint8)
    markers = find_marker(bits, marker=marker, max_errors=max_errors)
    extracted = []

    for item in markers:
        code_start = item.start_index + len(marker)
        code_end = code_start + code_len
        if code_end <= len(bits):
            code_bits = bits[code_start:code_end].copy()
            extracted.append(
                ExtractedCode(
                    marker_start=item.start_index,
                    marker_errors=item.errors,
                    code_start=code_start,
                    code_bits=code_bits,
                    code_str=bits_to_string(code_bits),
                )
            )

    return extracted
