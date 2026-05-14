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

# n=5, L=21
s = cutdown_de_bruijn(5, 21)
print(s)                       # → 1000110010100111...
print(verify_cutdown(s, 5))    # → is_valid: True

# n=9, L=412
s = cutdown_de_bruijn(9, 412)
print(s)
print(verify_cutdown(s, 9))    # → is_valid: True