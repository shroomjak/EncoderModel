import argparse
import sys
import time

import cv2
import numpy as np
import serial


class AutoRange:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.low = None
        self.high = None

    def update(self, values: np.ndarray):
        cur_low = float(np.percentile(values, 1))
        cur_high = float(np.percentile(values, 99))
        if self.low is None or self.high is None:
            self.low = cur_low
            self.high = cur_high
        else:
            self.low = (1 - self.alpha) * self.low + self.alpha * cur_low
            self.high = (1 - self.alpha) * self.high + self.alpha * cur_high
        if self.high - self.low < 1e-6:
            self.high = self.low + 1.0
        return self.low, self.high


def parse_serial_line(raw: bytes):
    try:
        text = raw.decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        parts = text.split(',')
        if len(parts) != 129:
            return None
        row_number = int(parts[0])
        pixels = np.array([int(x) for x in parts[1:]], dtype=np.float32)
        return row_number, pixels
    except Exception:
        return None


def normalize_pixels(pixels: np.ndarray, min_val, max_val, invert: bool):
    scaled = np.clip((pixels - min_val) / max(max_val - min_val, 1e-6), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    if invert:
        gray = 255 - gray
    return gray


def make_strip(gray: np.ndarray, pixel_width: int, strip_height: int):
    img = np.repeat(gray[np.newaxis, :], strip_height, axis=0)
    img = np.repeat(img, pixel_width, axis=1)
    return img


def build_argparser():
    p = argparse.ArgumentParser(
        description="Показывает 128 пикселей из serial-порта как серую полосу в реальном времени."
    )
    p.add_argument("--port", required=True, help="COM-порт, например COM3 или /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200, help="Скорость serial, по умолчанию 115200")
    p.add_argument("--pixel-width", type=int, default=6, help="Ширина одного пикселя на экране")
    p.add_argument("--strip-height", type=int, default=96, help="Высота полосы на экране")
    p.add_argument("--min", dest="min_val", type=float, default=None, help="Нижняя граница яркости")
    p.add_argument("--max", dest="max_val", type=float, default=None, help="Верхняя граница яркости")
    p.add_argument("--invert", action="store_true", help="Инвертировать яркость")
    p.add_argument("--overlay", action="store_true", help="Показывать номер строки и диапазон")
    return p


def main():
    args = build_argparser().parse_args()

    auto_range = args.min_val is None or args.max_val is None
    ranger = AutoRange(alpha=0.12) if auto_range else None

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        print(f"Не удалось открыть порт {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    time.sleep(1.0)

    win_name = "Serial grayscale strip"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    last_frame = np.zeros((args.strip_height, 128 * args.pixel_width), dtype=np.uint8)
    last_row = -1
    min_val = args.min_val
    max_val = args.max_val

    try:
        while True:
            packet = parse_serial_line(ser.readline())
            if packet is not None:
                last_row, pixels = packet
                if auto_range:
                    min_val, max_val = ranger.update(pixels)
                gray = normalize_pixels(pixels, min_val, max_val, args.invert)
                last_frame = make_strip(gray, args.pixel_width, args.strip_height)

                frame_bgr = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)

                if args.overlay:
                    text = f"row={last_row}  range=[{min_val:.1f}, {max_val:.1f}]"
                    cv2.putText(
                        frame_bgr,
                        text,
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imshow(win_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
