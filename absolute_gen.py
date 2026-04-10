#!/usr/bin/env python3
"""
Генератор абсолютного растра диска энкодера в DXF.

Структура диска:
code_1 - marker - code_2 - marker - ... - code_n - marker

Параметры:
- codes           : список десятичных кодов
- marker_code     : десятичный код репера
- code_bits       : число бит самого кода
- sector_bits     : общее число бит в секторе (код + репер)
- bit_width_mm    : дуговая ширина одного бита [мм]
- track_width_mm  : ширина дорожки [мм]
- n_segments      : число сегментов
- center_hole_mm  : диаметр центрального отверстия [мм]
- file_path       : имя выходного DXF

Принятая геометрия:
- bit = 1 -> внутренний радиус  (есть свет)
- bit = 0 -> внешний радиус     (нет света)
"""

import math
import ezdxf


CODES = [
    197	,
    314	,
    213	,
    298	,
    85	,
    426	,
    84	,
    427	,
    212	,
    299	,
    214	,
    297	,
    210	,
    301	,
    211	,
    300	,
    83	,
    428	,
    82	,
    429	,
    86	,
    425	,
    87	,
    424	,
    215	,
    296	,
    199	,
    312	,
    198	,
    313	,
    206	,
    307	,
    204	,
    51	,
    460	,
    179	,
    332	,
    163	,
    348	,
    167	,
    344	,
    165	,
    346	,
    164	,
    347	,
    180	,
    331	,
    181	,
    330	,
    53	,
    458	,
    52	,
    459	,
    54	,
    457	,
    50	,
    461	,
    178	,
    333	,
    182	,
    329	,
    166	,
    345	,
    230	,
    281	,
    228	,
    283	,
    229	,
    282	,
    101	,
    410	,
    100	,
    411	,
    116	,
    395	,
    117	,
    394	,
    119	,
    408	,
    103	,
    409	,
    102	,
    413	,
    99	,
    412	,
    227	,
    284	,
    231	,
    280	,
    183	,
    328	,
    55	,
    456	,
    311	,
    200	,
    279	,
    232	,
    277	,
    234	,
    405	,
    106	,
    404	,
    107	,
    276	,
    235	,
    278	,
    233	,
    406	,
    105	,
    402	,
    109	,
    403	,
    108	,
    407	,
    104	,
    471	,
    40	,
    467	,
    44	,
    339	,
    172	,
    338	,
    173	,
    466	,
    45	,
    470	,
    41	,
    468	,
    43	,
    469	,
    42	,
    453	,
    58	,
    325	,
    186	,
    341	,
    170	,
    340	,
    171	,
    342	,
    169	,
    326	,
    185	,
    306	,
]

MARKER = 8          #001000
SECTOR_BITS = 15
CODE_BITS = 9
BIT_WIDTH_MM = 0.8
TRACK_WIDTH_MM = 5
NUM_SEGMENTS = 55

def dec_to_bin(value, width):
    """Перевод десятичного числа в двоичную строку фиксированной длины."""
    return format(value, f"0{width}b")


def build_pattern(codes, marker_code, code_bits, sector_bits, n_segments):
    """
    Собирает полную битовую последовательность диска.
    В каждом секторе:
        [code_bits бит кода] + [оставшиеся биты репера]
    """
    marker_bits = sector_bits - code_bits

    used_codes = codes[:n_segments]

    pattern_str = ""
    for code in used_codes:
        code_word = dec_to_bin(code, code_bits)
        marker_word = dec_to_bin(marker_code, marker_bits)
        pattern_str += (code_word + marker_word)

    return [int(ch) for ch in pattern_str]


def generate_absolute_disc_dxf(
    codes,
    marker_code,
    code_bits,
    sector_bits,
    bit_width_mm,
    track_width_mm,
    n_segments,
    center_hole_mm=10.0,
    file_path="absolute_disc.dxf",
):
    """
    Генерация DXF абсолютного растра.
    Внешний радиус считается автоматически из общего числа бит.
    """
    pattern = build_pattern(codes, marker_code, code_bits, sector_bits, n_segments)
    print(pattern)

    total_bits = len(pattern)

    circumference_mm = total_bits * bit_width_mm
    r_outer_mm = circumference_mm / (2.0 * math.pi)
    r_inner_mm = r_outer_mm - track_width_mm

    doc = ezdxf.new()
    msp = doc.modelspace()

    angle_per_bit = 360.0 / total_bits

    for i, bit in enumerate(pattern):
        r_curr = r_inner_mm if bit == 1 else r_outer_mm

        a_start = i * angle_per_bit
        a_end = a_start + angle_per_bit

        # Перевод углов в систему DXF
        dxf_start = 90.0 - a_end
        dxf_end = 90.0 - a_start

        msp.add_arc(
            center=(0.0, 0.0),
            radius=r_curr,
            start_angle=dxf_start,
            end_angle=dxf_end,
        )

        # Радиальный переход на границе битов
        next_bit = pattern[(i + 1) % total_bits]
        r_next = r_inner_mm if next_bit == 1 else r_outer_mm

        if r_next != r_curr:
            rad = math.radians(90.0 - a_end)

            x1 = r_curr * math.cos(rad)
            y1 = r_curr * math.sin(rad)

            x2 = r_next * math.cos(rad)
            y2 = r_next * math.sin(rad)

            msp.add_line((x1, y1), (x2, y2))

    if center_hole_mm > 0:
        msp.add_circle(center=(0.0, 0.0), radius=center_hole_mm / 2.0)

    doc.saveas(file_path)

    print(f"[OK] -> {file_path}")
    print(f"[INFO] total_bits = {total_bits}")
    print(f"[INFO] r_outer_mm = {r_outer_mm:.6f}")
    print(f"[INFO] r_inner_mm = {r_inner_mm:.6f}")


if __name__ == "__main__":
    generate_absolute_disc_dxf(
        codes=CODES,
        marker_code=MARKER,
        code_bits=CODE_BITS,
        sector_bits=SECTOR_BITS,
        bit_width_mm=BIT_WIDTH_MM,
        track_width_mm=TRACK_WIDTH_MM,
        n_segments=NUM_SEGMENTS,
        center_hole_mm=0.0,
        file_path="absolute_disc.dxf",
    )

    '''
    generate_absolute_disc_dxf(
        codes=[3, 5, 6, 1, 7, 2],
        marker_code=1,
        code_bits=3,
        sector_bits=5,      # например: 3 бита кода + 2 бита репера
        bit_width_mm=5,
        track_width_mm=3.0,
        n_segments=12,
        center_hole_mm=20.0,
        file_path="absolute_disc_simple.dxf",
    )
    '''