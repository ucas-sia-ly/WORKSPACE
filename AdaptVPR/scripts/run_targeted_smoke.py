"""Five real masked edits plus zero/repeat/moved-mask controls; no planning/filtering."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageChops, ImageDraw

from generation.targeted_editor import (
    DiffusersMaskedEditor, MaskedEditorConfig, canonical_mask, file_sha256, load_task_pair, pixel_sha256,
)
from generation.targeted_inputs import read_targeted_tasks


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def difference(source, output):
    channels = ImageChops.difference(source.convert("RGB"), output.convert("RGB")).split()
    return ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])


def change_metrics(source, output, mask):
    diff = difference(source, output)
    changed = diff.point(lambda p: 255 if p else 0)
    selected = canonical_mask(mask, source.size)
    outside = ImageChops.invert(selected)
    inside_changed = ImageChops.multiply(changed, selected).histogram()[255]
    outside_changed = ImageChops.multiply(changed, outside).histogram()[255]
    area = selected.histogram()[255]
    inside_hist = ImageChops.multiply(diff, selected).histogram()
    return dict(mask_pixels=area, changed_pixels_inside=inside_changed, changed_pixels_outside=outside_changed,
                mean_max_channel_difference_inside=sum(i * n for i, n in enumerate(inside_hist)) / max(area, 1))


def save_edit(directory, source, mask, output, raw, audit):
    directory.mkdir(parents=True, exist_ok=False)
    source.save(directory / "source.png")
    mask.save(directory / "mask.png")
    canonical_mask(mask, source.size).save(directory / "mask_view.png")
    output.save(directory / "output.png")
    raw.save(directory / "raw_output.png")
    difference(source, output).save(directory / "difference.png")
    difference(source, raw).save(directory / "raw_difference.png")
    difference(source, output).point(lambda p: min(255, p * 4)).save(directory / "difference_x4.png")
    record = dict(audit=audit, output_metrics=change_metrics(source, output, mask),
                  raw_metrics=change_metrics(source, raw, mask))
    record["files_sha256"] = {p.name: file_sha256(p) for p in sorted(directory.glob("*.png"))}
    write_json(directory / "audit.json", record)
    return record


def contact_sheet(output_dir, directories):
    names = ["source.png", "mask_view.png", "raw_output.png", "output.png", "difference_x4.png"]
    labels = ["source", "mask (0/255 display)", "raw masked sampling", "output (+ RGB restoration)", "difference x4"]
    canvas = Image.new("RGB", (320 * len(names), 264 * len(directories)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, directory in enumerate(directories):
        for col, (name, label) in enumerate(zip(names, labels)):
            with Image.open(directory / name) as image:
                image = image.convert("RGB")
                image.thumbnail((320, 240))
                canvas.paste(image, (col * 320, row * 264 + 24))
            draw.text((col * 320 + 4, row * 264 + 4), f"{row + 1}: {label}", fill="black")
    canvas.save(output_dir / "contact_sheet.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="BoQ targets.jsonl (at least five records)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage3_targeted/smoke")
    parser.add_argument("--prompt", required=True, help="Fixed user-specified edit prompt; no planner")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true", help="Validate all five pairs/config without importing models")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed + 4 >= 2**63 or not args.prompt.strip():
        parser.error("Require nonempty prompt and seed in [0, 2**63-5)")
    config = MaskedEditorConfig.from_env()
    tasks = read_targeted_tasks(args.input, limit=5, seed=args.seed)
    if len(tasks) != 5:
        parser.error("Five-image smoke requires at least five records")
    pairs = [load_task_pair(task) for task in tasks]
    for source, _ in pairs:
        if source.width % 8 or source.height % 8:
            parser.error("Smoke source dimensions must be multiples of 8; no automatic resize")
    if args.check_only:
        print(json.dumps(dict(validated=5, generated=0, check_only=True)))
        return 0
    output_dir = args.output.resolve()
    if output_dir.exists():
        parser.error("Smoke output exists; select a fresh --output to preserve evidence")
    output_dir.mkdir(parents=True)
    summary = dict(schema_version=1, status="RUNNING", input_manifest=str(args.input.resolve()),
                   input_manifest_sha256=file_sha256(args.input), prompt=args.prompt, base_seed=args.seed,
                   note="real local masked sampling; not geometry/VPR/realism acceptance", samples=[], controls={})
    write_json(output_dir / "smoke_summary.json", summary)
    editor = DiffusersMaskedEditor(config)
    directories = []
    try:
        for index, (task, (source, mask)) in enumerate(zip(tasks, pairs)):
            seed = args.seed + index
            output = editor.edit(source, mask, args.prompt, seed)
            directory = output_dir / f"{index:02d}_{task['sample_id']}"
            record = save_edit(directory, source, mask, output, editor.last_raw_output, copy.deepcopy(editor.last_audit))
            directories.append(directory)
            if record["output_metrics"]["changed_pixels_inside"] == 0 or record["output_metrics"]["changed_pixels_outside"] != 0:
                raise RuntimeError("Smoke edit did not change inside mask exclusively")
            summary["samples"].append(dict(sample_id=task["sample_id"], image_key=task["image_key"],
                                            place_key=task["place_key"], stage2_metadata=task["stage2_metadata"],
                                            seed=seed, directory=directory.name, **record))
            write_json(output_dir / "smoke_summary.json", summary)
            print(f"Masked edit {index + 1}/5 verified and saved", flush=True)
        source, mask = pairs[0]
        # Same prompt/seed/source; changes in RAW outputs cannot be explained by final compositing.
        repeat = editor.edit(source, mask, args.prompt, args.seed)
        repeat_record = save_edit(output_dir / "controls/repeat", source, mask, repeat,
                                  editor.last_raw_output, copy.deepcopy(editor.last_audit))
        first_audit = summary["samples"][0]["audit"]
        repeat_ok = (pixel_sha256(repeat) == first_audit["output_pixel_sha256"] and
                     pixel_sha256(editor.last_raw_output) == first_audit["raw_output_pixel_sha256"] and
                     editor.last_audit["sampling_steps"] == first_audit["sampling_steps"])
        if not repeat_ok:
            raise RuntimeError("Same-seed repeat was not byte-identical in this environment")
        zero = Image.new("L", source.size, 0)
        identity = editor.edit(source, zero, args.prompt, args.seed)
        save_edit(output_dir / "controls/zero", source, zero, identity,
                  editor.last_raw_output, copy.deepcopy(editor.last_audit))
        if identity.tobytes() != source.tobytes() or editor.last_audit["generated"]:
            raise RuntimeError("Zero mask was not an identity operation")
        # Wrap translation is a diagnostic only, not a shape-matched scientific random control.
        shifted = ImageChops.offset(canonical_mask(mask, source.size), source.width // 2, source.height // 2)
        if shifted.tobytes() == canonical_mask(mask, source.size).tobytes():
            raise RuntimeError("Moved-mask diagnostic did not move the mask")
        moved = editor.edit(source, shifted, args.prompt, args.seed)
        moved_record = save_edit(output_dir / "controls/moved", source, shifted, moved,
                                 editor.last_raw_output, copy.deepcopy(editor.last_audit))
        moved_raw_differs = pixel_sha256(editor.last_raw_output) != first_audit["raw_output_pixel_sha256"]
        moved_latents_differ = editor.last_audit["sampling_steps"][-1]["latent_sha256"] != first_audit["sampling_steps"][-1]["latent_sha256"]
        if not moved_raw_differs or not moved_latents_differ or moved_record["output_metrics"]["changed_pixels_outside"]:
            raise RuntimeError("Moved mask did not affect raw sampling / edited support")
        summary["controls"] = dict(same_seed_raw_output_and_trace_identical=repeat_ok,
                                    zero_mask_identity_without_generation=True, moved_mask_changes_raw_output=moved_raw_differs,
                                    moved_mask_changes_sampling_latents=moved_latents_differ,
                                    moved_mask_translation=[source.width // 2, source.height // 2],
                                    moved_mask_is_scientific_matched_random_control=False)
        contact_sheet(output_dir, directories)
        summary.update(status="PASS", primary_image_count=5, nonzero_sampling_runs=7,
                       code_sha256={"generation/targeted_editor.py": file_sha256(ROOT / "generation/targeted_editor.py"),
                                    "scripts/run_targeted_smoke.py": file_sha256(Path(__file__))})
    except Exception as exc:
        summary.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output_dir / "smoke_summary.json", summary)
    print(json.dumps(dict(status=summary["status"], images=5, controls=summary["controls"], output=str(output_dir))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
