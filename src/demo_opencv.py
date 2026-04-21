import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import serial
import cv2
import numpy as np

from simulation import generate_ccd_signal, ADC_MAX
from signal_processing import process_signal, SignalProcessingResult
from marker_code import cells_to_bits, find_marker, extract_codes_after_markers


class AutoRange:
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.low: Optional[float] = None
        self.high: Optional[float] = None

    def update(self, values: np.ndarray) -> Tuple[float, float]:
        cur_low = float(np.percentile(values, 1))
        cur_high = float(np.percentile(values, 99))
        if self.low is None or self.high is None:
            self.low, self.high = cur_low, cur_high
        else:
            self.low = (1 - self.alpha) * self.low + self.alpha * cur_low
            self.high = (1 - self.alpha) * self.high + self.alpha * cur_high
        if self.high - self.low < 1e-6:
            self.high = self.low + 1.0
        return self.low, self.high


@dataclass
class FramePacket:
    row_index: int
    pixels: np.ndarray
    source_label: str


@dataclass
class Layout:
    width: int
    header_h: int
    top_h: int
    bottom_h: int
    x_range: Tuple[float, float]


@dataclass
class AnalysisResult:
    processed: SignalProcessingResult
    marker_spans: list
    code_spans: list
    code_text: str


def parse_csv_line(text: str) -> Optional[Tuple[int, np.ndarray]]:
    try:
        parts = text.strip().split(',')
        if len(parts) != 129:
            return None
        row_number = int(parts[0])
        pixels = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        return row_number, pixels
    except Exception:
        return None


def parse_serial_line(raw: bytes) -> Optional[Tuple[int, np.ndarray]]:
    return parse_csv_line(raw.decode('utf-8', errors='ignore'))


def normalize_pixels(pixels: np.ndarray, min_val: float, max_val: float, invert: bool = False) -> np.ndarray:
    scaled = np.clip((pixels - min_val) / max(max_val - min_val, 1e-6), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return 255 - gray if invert else gray


def make_layout(args, signal_len: int) -> Layout:
    width = signal_len * args.pixel_width
    header_h = args.header_height
    free_height = max(120, args.window_height - header_h)
    top_h = free_height // 2
    bottom_h = free_height - top_h
    x_range = (0.0, float(signal_len - 1))
    return Layout(width=width, header_h=header_h, top_h=top_h, bottom_h=bottom_h, x_range=x_range)


def x_mapper(x_range: Tuple[float, float], left_pad: int, plot_w: int):
    x_min, x_max = x_range

    def _map(x: float) -> int:
        t = (x - x_min) / (x_max - x_min)
        t = float(np.clip(t, 0.0, 1.0))
        return int(round(left_pad + t * (plot_w - 1)))

    return _map


def analyze_signal(pixels: np.ndarray, args) -> AnalysisResult:
    processed = process_signal(
        pixels,
        roi_margin=args.roi_margin,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
    )
    bits = cells_to_bits(
        processed.cell_values,
        threshold=args.bit_threshold,
    )
    markers = find_marker(
        bits,
        marker=args.marker,
        max_errors=args.marker_errors,
    )
    codes = extract_codes_after_markers(
        bits,
        marker=args.marker,
        code_len=args.code_len,
        max_errors=args.marker_errors,
    )

    centers = processed.centers
    pitch = processed.pitch
    half = pitch / 2.0

    marker_spans = []
    code_spans = []

    if len(centers) > 0:
        # Каждый бит занимает ячейку [centers[i] - half, centers[i] + half]
        # marker.start_index и code.code_start — индексы в массиве bits/centers
        for marker in markers:
            i0 = marker.start_index
            i1 = i0 + len(marker.matched_bits) - 1
            if i1 < len(centers):
                x1 = centers[i0] - half
                x2 = centers[i1] + half
                marker_spans.append((x1, x2))
        for code in codes:
            i0 = code.code_start
            i1 = i0 + len(code.code_bits) - 1
            if i1 < len(centers):
                x1 = centers[i0] - half
                x2 = centers[i1] + half
                code_spans.append((x1, x2))

    code_text = ', '.join(code.code_str for code in codes) if codes else '-'
    return AnalysisResult(
        processed=processed,
        marker_spans=marker_spans,
        code_spans=code_spans,
        code_text=code_text
    )


def draw_grayscale_panel(pixels: np.ndarray, min_val: float, max_val: float, invert: bool,
                         width: int, height: int, x_range: Tuple[float, float]) -> np.ndarray:
    gray = normalize_pixels(pixels, min_val, max_val, invert=invert)
    strip = np.repeat(gray[np.newaxis, :], height, axis=0)
    strip = np.repeat(strip, max(1, width // len(gray)), axis=1)
    if strip.shape[1] != width:
        strip = cv2.resize(strip, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)


def draw_binary_panel(processed, marker_spans, code_spans, width: int, height: int, x_range: Tuple[float, float]) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left_pad, right_pad, top_pad, bottom_pad = 0, 0, 0, 0
    plot_w = max(1, width - left_pad - right_pad)
    plot_h = max(1, height - top_pad - bottom_pad)
    x_map = x_mapper(x_range, left_pad, plot_w)

    y0 = top_pad + plot_h - 1
    y1 = top_pad + int(plot_h * 0.25)
    overlay = canvas.copy()

    for x1, x2 in marker_spans:
        cv2.rectangle(overlay, (x_map(x1), top_pad), (x_map(x2), top_pad + plot_h), (200, 220, 255), -1)
    for x1, x2 in code_spans:
        cv2.rectangle(overlay, (x_map(x1), top_pad), (x_map(x2), top_pad + plot_h), (210, 255, 210), -1)
    canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    centers = processed.centers
    pitch = processed.pitch
    # Границы ячеек: середины между соседними центрами
    edges = np.array([])
    if len(centers) >= 2:
        edges = (centers[1:] + centers[:-1]) / 2.0

    for e in edges:
        xx = x_map(float(e))
        cv2.line(canvas, (xx, top_pad), (xx, top_pad + plot_h), (210, 210, 210), 1, cv2.LINE_AA)

    for x in processed.roi:
        xx = x_map(float(x))
        cv2.line(canvas, (xx, top_pad), (xx, top_pad + plot_h), (0, 0, 180), 1, cv2.LINE_AA)

    pts = []
    for i, b in enumerate(processed.binary):
        pts.append((x_map(i), y1 if int(b) else y0))
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (left_pad, top_pad), (left_pad + plot_w, top_pad + plot_h), (180, 180, 180), 1)
    return canvas


def draw_header(packet: FramePacket, processed, code_text: str, marker_text: str,
                min_val: float, max_val: float, width: int, height: int) -> np.ndarray:
    header = np.full((height, width, 3), 245, dtype=np.uint8)
    lines = [
        f'source={packet.source_label} row={packet.row_index} range=[{min_val:.1f}, {max_val:.1f}]',
        f'roi={processed.roi} pitch={processed.pitch:.3f} phase={processed.phase:.3f}',
        f'marker={marker_text}',
        f'code={code_text}',
    ]
    y = 20
    for line in lines:
        cv2.putText(header, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)
        y += 20
    return header


def compose_frame(packet: FramePacket, args, ranger: AutoRange):
    pixels = np.asarray(packet.pixels, dtype=np.float32)
    min_val, max_val = (ranger.update(pixels) if args.min_val is None or args.max_val is None
                        else (args.min_val, args.max_val))

    _ = normalize_pixels(pixels, min_val, max_val, invert=args.invert)
    layout = make_layout(args, len(pixels))
    analysis = analyze_signal(pixels, args)

    header = draw_header(packet, analysis.processed, analysis.code_text, args.marker,
                         min_val, max_val, layout.width, layout.header_h)
    raw_panel = draw_grayscale_panel(pixels, min_val, max_val, args.invert, layout.width, layout.top_h, layout.x_range)
    binary_panel = draw_binary_panel(analysis.processed, analysis.marker_spans, analysis.code_spans,
                                     layout.width, layout.bottom_h, layout.x_range)

    frame = np.vstack([header, raw_panel, binary_panel])
    return frame, min_val, max_val


def simulation_packet(args) -> FramePacket:
    sim = generate_ccd_signal(
        code_bits=args.sim_code,
        marker=args.marker,
        noise_sigma=args.sim_noise_sigma,
        vignette_strength=args.sim_vignette,
        spread=args.sim_spread,
        bit_pixels=args.sim_bit_pixels,
        seed=args.sim_seed,
    )
    return FramePacket(row_index=0, pixels=np.asarray(sim['adc_noisy'], dtype=np.float32), source_label='simulation')


def serial_packets(args):
    if serial is None:
        raise RuntimeError('pyserial не установлен')
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(1.0)
    try:
        while True:
            packet = parse_serial_line(ser.readline())
            if packet is not None:
                row_index, pixels = packet
                yield FramePacket(row_index=row_index, pixels=pixels, source_label=f'serial:{args.port}')
    finally:
        ser.close()


def build_argparser():
    p = argparse.ArgumentParser(description='OpenCV-визуализация ПЗС-линейки: симуляция или serial CSV.')
    p.add_argument('--source', choices=['sim', 'serial'], default='sim')
    p.add_argument('--port')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--pixel-width', type=int, default=10)
    p.add_argument('--header-height', type=int, default=90)
    p.add_argument('--window-height', type=int, default=700)
    p.add_argument('--min', dest='min_val', type=float, default=None)
    p.add_argument('--max', dest='max_val', type=float, default=None)
    p.add_argument('--invert', action='store_true')
    p.add_argument('--marker', default='000100')
    p.add_argument('--code-len', type=int, default=9)
    p.add_argument('--marker-errors', type=int, default=0)
    p.add_argument('--bit-threshold', type=float, default=0.5)
    p.add_argument('--roi-margin', type=int, default=3)
    p.add_argument('--min-pitch', type=int, default=2)
    p.add_argument('--max-pitch', type=int, default=10)
    p.add_argument('--sim-code', default='101110011')
    p.add_argument('--sim-noise-sigma', type=float, default=500)
    p.add_argument('--sim-vignette', type=float, default=1.0)
    p.add_argument('--sim-spread', type=float, default=0.5)
    p.add_argument('--sim-bit-pixels', type=int, default=3)
    p.add_argument('--sim-seed', type=int, default=42)
    p.add_argument('--window', default='CCD demo OpenCV')
    return p


def main():
    args = build_argparser().parse_args()
    if args.source == 'serial' and not args.port:
        print('Для --source serial требуется --port', file=sys.stderr)
        sys.exit(2)

    ranger = AutoRange(alpha=0.12)
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    if args.source == 'sim':
        packet = simulation_packet(args)
        frame, min_val, max_val = compose_frame(packet, args, ranger)
        cv2.imshow(args.window, frame)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                break
    else:
        try:
            last_frame = np.full((args.window_height, 128 * args.pixel_width, 3), 240, dtype=np.uint8)
            for packet in serial_packets(args):
                try:
                    last_frame, min_val, max_val = compose_frame(packet, args, ranger)
                except Exception as e:
                    err = np.full_like(last_frame, 245)
                    cv2.putText(err, f'processing error: {e}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 180), 2, cv2.LINE_AA)
                    last_frame = err
                cv2.imshow(args.window, last_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
        except KeyboardInterrupt:
            pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
