import math
import ezdxf


CODE_BITS = 9
TRACK_WIDTH_MM = 5
INNER_RADIUS = 100.0

def generate_absolute_disc_dxf(
    pattern,
    r_inner_mm,
    track_width_mm,
    center_hole_mm=10.0,
    file_path="absolute_disc.dxf",
):
    """
    Генерация DXF абсолютного растра.
    Внешний радиус считается автоматически из общего числа бит.
    """
    total_bits = len(pattern)

    bit_width_mm = 2 * math.pi * (r_inner_mm + track_width_mm) / total_bits
    r_outer_mm = r_inner_mm + track_width_mm

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
    print(f"[INFO] track_width_mm = {track_width_mm:.6f}")
    print(f"[INFO] Bit width_mm = {bit_width_mm:.6f}")


def de_bruijn_full(n: int) -> str:
    """Полная B(2, n) через алгоритм Линдоновых слов."""
    a = [0] * 2 * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            if a[t - p] + 1 < 2:
                a[t] = a[t - p] + 1
                db(t + 1, t)

    db(1, 1)
    return "".join(map(str, sequence))


def cutdown_de_bruijn(n: int, L: int) -> str:
    """
    Cut-down де Брёйна длины L, порядок n.
    Все L циклических окон длины n — уникальны.
    """
    assert 1 <= L <= 2**n
    full = de_bruijn_full(n)
    if L == 2**n:
        return full

    # ищем подходящий сдвиг полной последовательности
    candidate = full + full
    for start in range(2**n):
        seg = candidate[start:start + L]
        ext = seg + seg[:n-1]
        windows = [ext[i:i+n] for i in range(L)]
        if len(set(windows)) == L:
            return seg

    raise ValueError(f"Не найдено для n={n}, L={L}")


def verify_cutdown(seq: str, n: int) -> dict:
    from collections import Counter
    L = len(seq)
    extended = seq + seq[:n - 1]
    windows = [extended[i:i+n] for i in range(L)]
    cnt = Counter(windows)

    return {
        "length": L,
        "unique_windows": len(cnt),
        "max_frequency": max(cnt.values()),
        "is_valid": len(cnt) == L and max(cnt.values()) == 1,
    }


if __name__ == "__main__":
    pattern = cutdown_de_bruijn(n=9, L=412)
    print(verify_cutdown(pattern, n=9))
    generate_absolute_disc_dxf(
        pattern=pattern,
        r_inner_mm=INNER_RADIUS,
        track_width_mm=TRACK_WIDTH_MM,
        center_hole_mm=0.0,
        file_path="absolute_de_bruijn_disc.dxf",
    )