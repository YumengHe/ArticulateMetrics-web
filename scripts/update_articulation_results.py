#!/usr/bin/env python3
"""Build the 10-case SPARK/URDFormer preview assets used by index.html."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

DEFAULT_DATA_ROOT = Path("/nas/yumenghe_shared/articulatemetrics_data")
DEFAULT_RENDERER = Path(
    "/data/yongfeishe/ArticulateMetrics/render/render_urdf_all_joints.py"
)
DEFAULT_PYTHON = Path(
    "/data/yongfeishe/miniconda3/envs/sparkprivate/bin/python"
)
# Each method maps to an ordered list of URDF path templates (relative to the data
# root); the first existing candidate wins. "{id}" expands to the case ID. URDFormer
# falls back to the preserved full-batch archive.
METHODS = {
    "spark": ["SPARK/{id}/mobility.urdf"],
    "urdformer": ["URDFormer/{id}/model.urdf", "URDFormer/_archive_full_100/{id}/model.urdf"],
    "artllm": ["ArtLLM/{id}/urdf/{id}.urdf"],
    "articulate": ["ArticulateAnything/{id}/final/mobility.urdf"],
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=repo / "articulation-results-cases.txt",
        help="Ordered case IDs; defaults to the checked-in showcase selection.",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--renderer", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output", type=Path, default=repo / "static/articulation-results")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_ids(path: Path, count: int) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    selected = ids[:count]
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError(f"Expected {count} unique IDs in {path}, got {len(selected)}")
    return selected


def resolve_urdf(data_root: Path, method: str, case_id: str) -> Path:
    """Resolve a method output from its ordered path templates."""
    candidates = [data_root / tmpl.format(id=case_id) for tmpl in METHODS[method]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    locations = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing {method} URDF for {case_id}; checked: {locations}")


def joint_count(urdf: Path) -> int:
    root = ET.parse(urdf).getroot()
    return sum(joint.get("type") in {"revolute", "continuous", "prismatic"} for joint in root.findall("joint"))


def normalize_gif(source: Path, destination: Path, size: int, fps: float) -> None:
    image = Image.open(source)
    frames: list[Image.Image] = []
    boxes = []
    try:
        while True:
            frame = image.convert("RGB")
            frames.append(frame.copy())
            background = Image.new("RGB", frame.size, frame.getpixel((0, 0)))
            difference = ImageChops.difference(frame, background).convert("L")
            foreground = difference.point(lambda value: 255 if value > 8 else 0)
            bbox = foreground.getbbox()
            if bbox:
                boxes.append(bbox)
            image.seek(image.tell() + 1)
    except EOFError:
        pass

    if not boxes:
        raise ValueError(f"Rendered GIF is blank: {source}")

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    padding = max(8, round(max(right - left, bottom - top) * 0.08))
    crop = (
        max(0, left - padding),
        max(0, top - padding),
        min(frames[0].width, right + padding),
        min(frames[0].height, bottom + padding),
    )

    normalized = []
    for frame in frames:
        fitted = ImageOps.contain(frame.crop(crop), (size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
        normalized.append(canvas)

    destination.parent.mkdir(parents=True, exist_ok=True)
    palette = normalized[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in normalized]
    quantized[0].save(
        destination,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=1,
        optimize=False,
    )


def render_one(
    method: str, case_id: str, source: Path, destination: Path, args: argparse.Namespace
) -> str:
    if destination.exists() and not args.force:
        return f"skip {method}/{case_id}"

    with tempfile.TemporaryDirectory(prefix=f"articulation-{method}-{case_id}-") as temp:
        if joint_count(source) == 0:
            rendered = Path(temp) / f"{source.stem}_static.png"
            command = [
                str(args.python),
                str(args.renderer.with_name("render_front_image.py")),
                "--urdf",
                str(source),
                "--output",
                str(rendered),
                "--image-size",
                str(args.image_size),
                "--up-axis",
                "z",
            ]
        else:
            rendered = Path(temp) / f"{source.stem}_animation.gif"
            command = [
                str(args.python),
                str(args.renderer),
                "--urdf",
                str(source),
                "--output",
                temp,
                "--frames",
                str(args.frames),
                "--fps",
                str(args.fps),
                "--image-size",
                str(args.image_size),
                "--up-axis",
                "z",
            ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                f"Render failed for {method}/{case_id}:\n{result.stdout}{result.stderr}"
            )
        normalize_gif(rendered, destination, args.image_size, args.fps)
    return f"rendered {method}/{case_id}"


def write_manifest(output: Path, items: list[dict]) -> None:
    payload = json.dumps(items, indent=2, ensure_ascii=True)
    (output / "results.js").write_text(
        "// Generated by scripts/update_articulation_results.py.\n"
        f"window.ARTICULATION_RESULTS = {payload};\n"
    )


def main() -> None:
    args = parse_args()
    ids_file = args.ids_file or args.data_root / "eval_100.txt"
    case_ids = load_ids(ids_file, args.count)
    args.output.mkdir(parents=True, exist_ok=True)

    tasks = []
    items = []
    for case_id in case_ids:
        gt_dir = args.data_root / "GT" / case_id
        input_image = gt_dir / "front.png"
        meta_path = gt_dir / "meta.json"
        if not input_image.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing GT input or metadata for {case_id}")
        category = json.loads(meta_path.read_text()).get("model_cat", "Unknown")
        case_output = args.output / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_image, case_output / "input.png")

        item = {"id": case_id, "category": category, "methods": {}}
        for key in METHODS:
            urdf = resolve_urdf(args.data_root, key, case_id)
            item["methods"][key] = {"movableJoints": joint_count(urdf)}
            tasks.append((key, case_id, urdf, case_output / f"{key}.gif"))
        items.append(item)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(render_one, *task, args) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)

    write_manifest(args.output, items)
    print(f"Updated {len(items)} cases in {args.output}")


if __name__ == "__main__":
    main()
