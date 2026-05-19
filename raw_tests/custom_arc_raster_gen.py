import math
import ezdxf

# ──────────────────────────────────────────────
# ПАРАМЕТРЫ (задаются пользователем)
# ──────────────────────────────────────────────
PATTERN        = "10" * 30 + "1"     # строго заданная последовательность из 0 и 1
BIT_WIDTH_MM   = 2                   # ширина одного бита по внешнему радиусу, мм
CODE_WIDTH_MM  = 5.0                 # радиальная ширина дорожки (от r_inner до r_outer), мм
INNER_RADIUS   = 100.0               # внешний радиус, мм
CENTER_HOLE_MM = 0.0                 # отверстие по центру (0 — без отверстия)
OUTPUT_FILE    = "custom_arc_raster.dxf"
# ──────────────────────────────────────────────


def generate_arc_raster_dxf(
    pattern: str,
    bit_width_mm: float,
    code_width_mm: float,
    r_inner_mm: float,
    center_hole_mm: float = 0.0,
    file_path: str = "custom_arc_raster.dxf",
) -> None:
    """
    Генерация DXF растра для строго заданной битовой последовательности.

    Принцип кодирования (аналогично де Брёйновскому растру):
      - бит '1' → дуга по r_outer (выступ, максимальный радиус)
      - бит '0' → дуга по r_inner (впадина, минимальный радиус)

    Угол последовательности НЕ фиксируется в 360°, а определяется
    автоматически из заданной ширины бита по внешнему радиусу:
        angle_per_bit = bit_width_mm / r_outer_mm  [рад]

    Начало — 90° (12 часов), отсчёт по часовой стрелке.

    Parameters
    ----------
    pattern       : строка из '0' и '1'
    bit_width_mm  : ширина одного бита по внешнему радиусу, мм
    code_width_mm : радиальная ширина дорожки (r_outer - r_inner), мм
    r_inner_mm    : внутренний радиус, мм
    center_hole_mm: диаметр центрального отверстия (0 = нет)
    file_path     : путь для сохранения .dxf файла
    """
    assert len(pattern) > 0, "Последовательность не может быть пустой"
    assert all(c in "01" for c in pattern), "Паттерн должен содержать только '0' и '1'"
    assert bit_width_mm > 0, "Ширина бита должна быть положительной"
    assert code_width_mm > 0, "Ширина кода должна быть положительной"
    assert r_inner_mm > 0, "r_inner должен быть больше code_width"

    total_bits = len(pattern)
    r_outer_mm = r_inner_mm + code_width_mm

    # Угол на один бит (радианы и градусы)
    angle_per_bit_rad = bit_width_mm / r_outer_mm
    angle_per_bit_deg = math.degrees(angle_per_bit_rad)
    total_angle_deg   = angle_per_bit_deg * total_bits

    doc = ezdxf.new()
    msp = doc.modelspace()

    # Начало последовательности — 90° (12 часов), движение по часовой стрелке
    # В системе DXF углы — против часовой стрелки, поэтому:
    # DXF_angle = 90° - физический_угол_от_начала

    for i, bit in enumerate(pattern):
        r_curr = r_outer_mm if bit == "1" else r_inner_mm

        # Физические углы от начала (по часовой стрелке), градусы
        phys_start = i * angle_per_bit_deg
        phys_end   = phys_start + angle_per_bit_deg

        # Перевод в DXF (против часовой стрелки от оси X)
        dxf_start = 90.0 - phys_end
        dxf_end   = 90.0 - phys_start

        msp.add_arc(
            center=(0.0, 0.0),
            radius=r_curr,
            start_angle=dxf_start,
            end_angle=dxf_end,
        )

        # Радиальный переход на ПРАВОЙ границе бита (конец текущего бита)
        # Рисуется только если следующий бит меняет радиус
        is_last = (i == total_bits - 1)
        if not is_last:
            next_bit = pattern[i + 1]
            r_next = r_outer_mm if next_bit == "1" else r_inner_mm
        else:
            r_next = None  # правая граница последнего бита — просто торец

        # Правая граница текущего бита
        rad_right = math.radians(90.0 - phys_end)
        x_right_outer = r_outer_mm * math.cos(rad_right)
        y_right_outer = r_outer_mm * math.sin(rad_right)
        x_right_inner = r_inner_mm * math.cos(rad_right)
        y_right_inner = r_inner_mm * math.sin(rad_right)

        if r_next is not None and r_next != r_curr:
            # Радиальная черта на переходе между битами с разными уровнями
            x1 = r_curr * math.cos(rad_right)
            y1 = r_curr * math.sin(rad_right)
            x2 = r_next * math.cos(rad_right)
            y2 = r_next * math.sin(rad_right)
            msp.add_line((x1, y1), (x2, y2))

        # Для первого бита — левый торец (вертикальная радиальная черта)
        if i == 0:
            rad_left = math.radians(90.0 - phys_start)
            x_left_outer = r_outer_mm * math.cos(rad_left)
            y_left_outer = r_outer_mm * math.sin(rad_left)
            x_left_inner = r_inner_mm * math.cos(rad_left)
            y_left_inner = r_inner_mm * math.sin(rad_left)
            msp.add_line(
                (x_left_inner, y_left_inner),
                (x_left_outer, y_left_outer),
            )

        # Для последнего бита — правый торец
        if is_last:
            msp.add_line(
                (x_right_inner, y_right_inner),
                (x_right_outer, y_right_outer),
            )

    # Центральное отверстие
    if center_hole_mm > 0:
        msp.add_circle(center=(0.0, 0.0), radius=center_hole_mm / 2.0)

    doc.saveas(file_path)

    # Сводка
    print(f"[OK]   Файл сохранён -> {file_path}")
    print(f"[INFO] Паттерн       : {pattern}")
    print(f"[INFO] Длина         : {total_bits} бит")
    print(f"[INFO] r_outer       = {r_outer_mm:.4f} мм")
    print(f"[INFO] r_inner       = {r_inner_mm:.4f} мм")
    print(f"[INFO] code_width    = {code_width_mm:.4f} мм")
    print(f"[INFO] bit_width     = {bit_width_mm:.4f} мм  (по внешнему радиусу)")
    print(f"[INFO] angle/bit     = {angle_per_bit_deg:.6f}°")
    print(f"[INFO] total_angle   = {total_angle_deg:.6f}°")
    print(f"[INFO] arc_length    = {bit_width_mm * total_bits:.4f} мм  (по внешнему радиусу)")


if __name__ == "__main__":
    generate_arc_raster_dxf(
        pattern=PATTERN,
        bit_width_mm=BIT_WIDTH_MM,
        code_width_mm=CODE_WIDTH_MM,
        r_inner_mm=INNER_RADIUS,
        center_hole_mm=CENTER_HOLE_MM,
        file_path=OUTPUT_FILE,
    )
