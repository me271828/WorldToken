"""Render a 3x2 grid video from a RoboCasa diffusion-action holdout prediction trace.

Reads one H5 written by
``diffusion_wm.training.holdout.write_robocasa_holdout_prediction_trace``
(e.g. ``<output_dir>/holdout_prediction_traces/holdout_step_00001000.h5``) and writes
an MP4 with a 3x2 panel grid. RoboCasa has three cameras, and for each camera
the trace stores two image streams, so every frame shows six panels:

    rows    = the three cameras (in /image_keys order)
    columns = [ observed , predicted ]

Stream meaning (see the trace's /meta/json "alignment" field):

* observed        ``rgb/<cam>[t]``                 -- the real observation at t.
* predicted       ``rgb_predicted/<cam>[t]`` -- the sampled-action
  conditioned decode produced at t-1 for t. Frame 0 has no previous
  prediction, so its predicted column is black (warmup).

Use ``--layout camera-cols`` to transpose (cameras across columns, the three
stream types down rows).

CLI:
    python -m decoder.make_robocasa_holdout_grid_video <h5_path> \\
        [--out path.mp4] [--fps 5] [--start 0] [--stop N] [--scale 2] \\
        [--gap 4] [--layout camera-rows|camera-cols] [--no-text] \\
        [--font /path] [--font-size 18] [--include-invalid] \\
        [--warmup-frames N | --no-warmup-mask]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "imageio (with the ffmpeg plugin) is required: pip install 'imageio[ffmpeg]'"
    ) from exc


_COLUMN_LABELS = ("observed", "predicted")
_COLUMN_SHORT = {"observed": "obs", "predicted": "pred"}
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/simhei.ttf",
)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _short_cam(key: str) -> str:
    """Compact a RoboCasa camera key, e.g. robot0_agentview_left_image -> agentview_left."""
    name = key
    for prefix in ("robot0_", "robot1_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith("_image"):
        name = name[: -len("_image")]
    return name


def _load_font(font_path: Path | None, font_size: int):
    try:
        from PIL import ImageFont
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required for text overlay; install pillow or use --no-text") from exc
    candidates = []
    if font_path is not None:
        candidates.append(Path(font_path))
    candidates.extend(Path(p) for p in _FONT_CANDIDATES)
    for candidate in candidates:
        if candidate.expanduser().exists():
            return ImageFont.truetype(str(candidate.expanduser()), font_size)
    return ImageFont.load_default()


def _draw_label(frame: np.ndarray, label: str, font, font_size: int, *, anchor: str = "tl") -> np.ndarray:
    """Draw a small labelled chip on `frame`. anchor in {tl, tr, bl, br}."""
    if not label or font is None:
        return frame
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad_x = max(3, int(font_size * 0.30))
    pad_y = max(2, int(font_size * 0.22))
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box_w, box_h = text_w + 2 * pad_x, text_h + 2 * pad_y
    img_w, img_h = image.size
    ox = 0 if anchor in ("tl", "bl") else max(0, img_w - box_w)
    oy = 0 if anchor in ("tl", "tr") else max(0, img_h - box_h)
    draw.rectangle((ox, oy, ox + box_w, oy + box_h), fill=(0, 0, 0, 170))
    draw.text((ox + pad_x - bbox[0], oy + pad_y - bbox[1]), label, fill=(255, 255, 255, 255), font=font)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _scale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return frame
    # Nearest-neighbour upscale keeps quantization/blur honest (bilinear would
    # smear prediction artefacts and make obs vs. pred look falsely similar).
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def _hstack(panels: list[np.ndarray], gap_px: int, sep_rgb=(32, 32, 32)) -> np.ndarray:
    if gap_px <= 0:
        return np.concatenate(panels, axis=1)
    h, _, c = panels[0].shape
    sep = np.empty((h, gap_px, c), dtype=panels[0].dtype)
    sep[:] = np.asarray(sep_rgb, dtype=panels[0].dtype)
    parts: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        if idx > 0:
            parts.append(sep)
        parts.append(panel)
    return np.concatenate(parts, axis=1)


def _vstack(rows: list[np.ndarray], gap_px: int, sep_rgb=(32, 32, 32)) -> np.ndarray:
    if gap_px <= 0:
        return np.concatenate(rows, axis=0)
    _, w, c = rows[0].shape
    sep = np.empty((gap_px, w, c), dtype=rows[0].dtype)
    sep[:] = np.asarray(sep_rgb, dtype=rows[0].dtype)
    parts: list[np.ndarray] = []
    for idx, row in enumerate(rows):
        if idx > 0:
            parts.append(sep)
        parts.append(row)
    return np.concatenate(parts, axis=0)


def _compose_grid(cells: list[list[np.ndarray]], gap_px: int) -> np.ndarray:
    return _vstack([_hstack(row, gap_px) for row in cells], gap_px)


def _pad_to_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if not (pad_h or pad_w):
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _add_header(grid: np.ndarray, text: str, font, font_size: int, bar_h: int) -> np.ndarray:
    if font is None or not text:
        return grid
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=grid.dtype)
    bar = _draw_label(bar, text, font, font_size, anchor="tl")
    return np.concatenate([bar, grid], axis=0)


def _resolve_warmup_mask(f: h5py.File, total: int, warmup_frames: int | None, disable: bool) -> np.ndarray:
    if disable:
        return np.zeros((total,), dtype=bool)
    if warmup_frames is not None:
        mask = np.zeros((total,), dtype=bool)
        mask[: max(0, min(int(warmup_frames), total))] = True
        return mask
    if "diag/is_warmup" in f:
        return np.asarray(f["diag/is_warmup"][:], dtype=bool)
    mask = np.zeros((total,), dtype=bool)
    if total > 0:
        mask[0] = True
    return mask


def make_grid_video(
    h5_path: Path,
    out_path: Path | None,
    *,
    fps: int,
    start: int,
    stop: int | None,
    scale: int,
    gap_px: int,
    layout: str,
    overlay_text: bool,
    font_path: Path | None,
    font_size: int | None,
    include_invalid: bool,
    warmup_frames: int | None,
    disable_warmup_mask: bool,
) -> None:
    if scale < 1:
        raise ValueError(f"--scale must be >= 1, got {scale}")
    if gap_px < 0:
        raise ValueError(f"--gap must be >= 0, got {gap_px}")
    if layout not in ("camera-rows", "camera-cols"):
        raise ValueError(f"--layout must be camera-rows or camera-cols, got {layout!r}")
    if out_path is None:
        out_path = h5_path.with_suffix(".grid.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if "image_keys" in f:
            image_keys = [_decode(x) for x in f["image_keys"][:]]
        elif "rgb" in f:
            image_keys = sorted(f["rgb"].keys())
        else:
            raise KeyError(f"{h5_path} has no /image_keys or /rgb group — not a RoboCasa holdout trace")
        if not image_keys:
            raise ValueError(f"{h5_path} lists no cameras")

        def _has_stream(group: str) -> bool:
            return all(f"{group}/{cam}" in f for cam in image_keys)

        for required in ("rgb", "rgb_predicted"):
            if not _has_stream(required):
                raise KeyError(f"{h5_path} missing required image stream {required!r}/<cam>")
        columns = [
            (lbl, grp)
            for (lbl, grp) in (
                ("observed", "rgb"),
                ("predicted", "rgb_predicted"),
            )
            if lbl in ("observed", "predicted") or _has_stream(grp)
        ]
        col_labels = [lbl for (lbl, _grp) in columns]
        streams = {lbl: grp for (lbl, grp) in columns}

        ref = f[f"rgb/{image_keys[0]}"]
        total = int(ref.shape[0])
        panel_h = int(ref.shape[1])
        start = max(0, int(start))
        stop = total if stop is None else min(total, int(stop))
        if not (0 <= start < stop <= total):
            raise ValueError(f"invalid frame range [{start}, {stop}) for length {total}")

        valid = (
            np.asarray(f["valid_mask"][:], dtype=bool)
            if "valid_mask" in f
            else np.ones((total,), dtype=bool)
        )
        warmup_mask = _resolve_warmup_mask(f, total, warmup_frames, disable_warmup_mask)
        frame_index = (
            np.asarray(f["frame_index"][:]) if "frame_index" in f else np.arange(total)
        )
        meta = json.loads(_decode(f["meta/json"][()])) if "meta/json" in f else {}
        step = int(meta.get("global_step", 0))
        lang = str(meta.get("lang", ""))
        pred_mse = {cam: _read_diag(f, f"diag/pred_image_mse/{cam}", total) for cam in image_keys}

        if font_size is None:
            font_size = max(11, int(panel_h * scale * 0.11))
        font = _load_font(font_path, font_size) if overlay_text else None
        bar_h = font_size + 2 * max(2, int(font_size * 0.22)) if font is not None else 0

        # Preload all frames into memory (a single holdout trace is small).
        data = {
            label: {cam: np.asarray(f[f"{group}/{cam}"][:]) for cam in image_keys}
            for label, group in streams.items()
        }

    rendered = 0
    skipped_invalid = 0
    with imageio.get_writer(
        str(out_path),
        format="FFMPEG",
        fps=fps,
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for t in range(start, stop):
            if not include_invalid and not bool(valid[t]):
                skipped_invalid += 1
                continue
            is_warmup = bool(warmup_mask[t])

            # cell_by[cam][col_label] -> labelled, scaled panel
            cell_by: dict[str, dict[str, np.ndarray]] = {}
            for cam in image_keys:
                short = _short_cam(cam)
                col_cells: dict[str, np.ndarray] = {}
                for col in col_labels:
                    raw = data[col][cam][t]
                    if col == "predicted" and is_warmup:
                        raw = np.zeros_like(raw)
                    panel = _scale_frame(raw, scale)
                    if font is not None:
                        # Camera name rides on the observed panel only (it labels
                        # the whole row/column); prediction panels keep just the
                        # short type tag + an MSE chip, so nothing overlaps.
                        if col == "observed":
                            tl_label = f"{short} | obs"
                        elif col == "predicted" and is_warmup:
                            tl_label = "pred(warmup)"
                        else:
                            tl_label = _COLUMN_SHORT[col]
                        panel = _draw_label(panel, tl_label, font, font_size, anchor="tl")
                        if col == "predicted":
                            panel = _draw_label(panel, _fmt_mse(pred_mse[cam][t]), font, font_size, anchor="tr")
                    col_cells[col] = panel
                cell_by[cam] = col_cells

            if layout == "camera-rows":
                cells = [[cell_by[cam][col] for col in col_labels] for cam in image_keys]
            else:  # camera-cols: rows = stream types, columns = cameras
                cells = [[cell_by[cam][col] for cam in image_keys] for col in col_labels]

            grid = _compose_grid(cells, gap_px=gap_px * scale)
            if font is not None:
                header = f"step {step}  frame {int(frame_index[t])}  t={t}"
                if lang:
                    header += f"  |  {lang}"
                grid = _add_header(grid, header, font, font_size, bar_h)
            grid = _pad_to_even(np.ascontiguousarray(grid))
            writer.append_data(grid)
            rendered += 1

    print(f"saved: {out_path}")
    print(f"cameras: {', '.join(_short_cam(k) for k in image_keys)}")
    n_cam, n_col = len(image_keys), len(col_labels)
    grid_dims = f"{n_cam}x{n_col}" if layout == "camera-rows" else f"{n_col}x{n_cam}"
    print(f"layout: {layout}  (grid = {grid_dims} = {n_cam * n_col} panels; columns = {', '.join(col_labels)})")
    print(f"frames: rendered={rendered} skipped_invalid={skipped_invalid} range=[{start},{stop}) at {fps} fps")


def _read_diag(f: h5py.File, key: str, total: int) -> np.ndarray:
    if key in f:
        return np.asarray(f[key][:], dtype=np.float32)
    return np.full((total,), np.nan, dtype=np.float32)


def _fmt_mse(value: float) -> str:
    v = float(value)
    if not np.isfinite(v):
        return "mse n/a"
    return f"mse {v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 3x2 (camera x {observed, predicted}) grid MP4 from a RoboCasa holdout trace.",
    )
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--scale", type=int, default=2, help="Integer nearest-neighbour upscale per panel. Default 2.")
    parser.add_argument("--gap", type=int, default=4, help="Separator width in panel pixels (multiplied by --scale).")
    parser.add_argument(
        "--layout",
        choices=("camera-rows", "camera-cols"),
        default="camera-rows",
        help="camera-rows: rows=cameras, cols=[observed,predicted] (default). camera-cols transposes.",
    )
    parser.add_argument("--no-text", action="store_true", help="Disable all label/header overlays.")
    parser.add_argument("--font", type=Path, default=None, help="TrueType/OpenType font path for labels.")
    parser.add_argument("--font-size", type=int, default=None)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Render padded/invalid frames too (default: skip frames where /valid_mask is False).",
    )
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--warmup-frames",
        type=int,
        default=None,
        help="Force the first N frames to black out the predicted column (overrides /diag/is_warmup).",
    )
    warmup_group.add_argument(
        "--no-warmup-mask",
        action="store_true",
        help="Disable warmup masking; show rgb_predicted for every frame (incl. the t=0 placeholder).",
    )
    args = parser.parse_args()
    make_grid_video(
        args.h5_path,
        args.out,
        fps=args.fps,
        start=args.start,
        stop=args.stop,
        scale=args.scale,
        gap_px=args.gap,
        layout=args.layout,
        overlay_text=not args.no_text,
        font_path=args.font,
        font_size=args.font_size,
        include_invalid=args.include_invalid,
        warmup_frames=args.warmup_frames,
        disable_warmup_mask=args.no_warmup_mask,
    )


if __name__ == "__main__":
    main()
