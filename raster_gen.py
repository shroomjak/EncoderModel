#!/usr/bin/env python3
"""
Incremental encoder raster generator for CAD (Компас 3D).

Выходной файл SVG содержит один замкнутый контур — прямоугольный меандр
вдоль окружности диска. Используется как эскиз для вытягивания в 3D.

Геометрия:
  bit = 1 (прозрачный, дырка) → меандр уходит на r_inner
  bit = 0 (непрозрачный, сплошной) → меандр идёт по r_outer

  r_outer = внешний радиус диска (задаётся как параметр)
  r_inner = r_outer - track_width

Контур строится SVG-командами A (дуга) + L (радиальный скачок),
то есть дуги математически точные, не полигональные.

Дополнительно отрисовываются пунктирными линиями:
  - граница r_inner (внутренний край дорожки)
  - центральное отверстие (опционально)

Параметры generate_disc_meander:
  r_outer_mm     : внешний радиус диска [мм]
  track_width_mm : ширина кодовой дорожки (радиальная) [мм]
  bit_width_mm   : желаемая дуговая ширина одного бита [мм]
  k_ones         : число бит '1' в блоке (дырки)
  l_zeros        : число бит '0' в блоке (сплошные)
  center_hole_mm : диаметр центрального отверстия (только контур, мм)
  file_path      : путь к выходному SVG-файлу
"""

import math
import ezdxf

def choose_n_bits(
    r_outer_mm: float,
    bit_width_mm: float,
    k_ones: int,
    l_zeros: int,
) -> tuple:
    """Подбирает N_bits, кратное (k+l), с arc/bit ≈ bit_width_mm.

    Возвращает (N_bits, arc_per_bit_mm, n_blocks).
    """
    block = k_ones + l_zeros
    if block <= 0:
        raise ValueError("k_ones + l_zeros должно быть > 0")
    if bit_width_mm <= 0:
        raise ValueError("bit_width_mm должно быть > 0")

    circumference = 2.0 * math.pi * r_outer_mm
    n_ideal = circumference / bit_width_mm

    # Ближайшее кратное block к n_ideal, перебираем окрестность ±100 блоков
    base = max(1, round(n_ideal / block)) * block
    best = None

    for delta in range(-100, 101):
        n = base + delta * block
        if n <= 0:
            continue
        arc = circumference / n
        err = abs(arc - bit_width_mm) / bit_width_mm
        if best is None or err < best[0]:
            best = (err, n, arc, n // block)

    rel_err, n_bits, arc_per_bit, n_blocks = best

    if rel_err > 0.05:
        print(
            f"[WARN] arc/bit={arc_per_bit:.4f} мм отличается от "
            f"bit_width={bit_width_mm:.4f} мм на {rel_err*100:.1f}%. "
            f"Подбери другой bit_width_mm или r_outer_mm."
        )
    else:
        print(
            f"[INFO] N_bits={n_bits}, arc/bit={arc_per_bit:.4f} мм "
            f"(цель {bit_width_mm:.4f} мм, ошибка {rel_err*100:.2f}%), "
            f"блоков={n_blocks}"
        )

    return n_bits, arc_per_bit, n_blocks


def generate_disc_meander_dxf(
    r_outer_mm, track_width_mm, bit_width_mm,
    k_ones, l_zeros,
    center_hole_mm=10.0,
    file_path="disc_meander.dxf"
):
    n_bits, arc_per_bit, n_blocks = choose_n_bits(
        r_outer_mm, bit_width_mm, k_ones, l_zeros
    )
    r_out = r_outer_mm
    r_in  = r_outer_mm - track_width_mm
    pattern = ([1]*k_ones + [0]*l_zeros) * n_blocks

    if center_hole_mm >= 2 * r_out:
        raise ValueError(
            f"Диаметр отверстия {center_hole_mm} мм >= диаметра диска {2 * r_out} мм"
        )

    doc = ezdxf.new()
    msp = doc.modelspace()

    angle_per_bit = 360.0 / n_bits

    for i, bit in enumerate(pattern):
        r_curr = r_out if bit == 0 else r_in
        a_start = i * angle_per_bit
        a_end   = a_start + angle_per_bit

        # DXF: углы в градусах, от оси X, против часовой стрелки
        # Наш 0° — сверху (−90° в DXF), направление — по часовой,
        # поэтому: dxf_angle = 90 − our_angle
        dxf_start = 90.0 - a_end    # конец нашего бита = начало дуги DXF
        dxf_end   = 90.0 - a_start  # начало нашего бита = конец дуги DXF

        msp.add_arc(
            center=(0, 0),
            radius=r_curr,
            start_angle=dxf_start,
            end_angle=dxf_end,
        )

        # Радиальный скачок на границе битов
        r_next = r_out if pattern[(i+1) % n_bits] == 0 else r_in
        if r_next != r_curr:
            rad = math.radians(90.0 - a_end)
            x = r_curr * math.cos(rad)
            y = r_curr * math.sin(rad)
            x2 = r_next * math.cos(rad)
            y2 = r_next * math.sin(rad)
            msp.add_line((x, y), (x2, y2))

    # Центральное отверстие — точная окружность
    if center_hole_mm > 0:
        msp.add_circle(center=(0, 0), radius=center_hole_mm / 2)

    doc.saveas(file_path)
    print(f"[OK] → {file_path}")


# ─────────────────────────────────────────────────────────────────────
# Примеры: серия для тестирования 3D-печати
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    R_OUTER    = 60.0   # внешний радиус диска, мм
    TRACK      = 3.0    # ширина дорожки, мм
    HOLE_DIAM  = 60.0   # диаметр центрального отверстия, мм

    # Скважность 1:1 (равные дырки и сплошные), разные ширины бита
    for bw in [0.30, 0.40, 0.50, 0.60, 5]:
        generate_disc_meander_dxf(
            r_outer_mm     = R_OUTER,
            track_width_mm = TRACK,
            bit_width_mm   = bw,
            k_ones         = 1,
            l_zeros        = 1,
            center_hole_mm = HOLE_DIAM,
            file_path=f"disc_R{int(R_OUTER)}_bit{str(bw).replace('.', 'p')}_1on1off.dxf",
        )

    """
    print()

     # Скважность 2:2 при фиксированной ширине бита
     generate_disc_meander(
         r_outer_mm     = R_OUTER,
         track_width_mm = TRACK,
         bit_width_mm   = 0.50,
         k_ones         = 2,
         l_zeros        = 2,
         center_hole_mm = HOLE_DIAM,
         file_path      = f"disc_R{int(R_OUTER)}_bit0p50_2on2off.svg",
     )
     """
