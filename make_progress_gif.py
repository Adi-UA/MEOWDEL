import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs", nargs="+", help="one or more run dirs, e.g. outputs/20260708_104330 outputs/20260708_121740"
    )
    parser.add_argument("--labels", nargs="+", help="one label per run_dir; defaults to each run dir's folder name")
    parser.add_argument("--output", default="assets/training_progress.gif")
    parser.add_argument("--duration-ms", type=int, default=200)
    parser.add_argument("--hold-first-ms", type=int, default=2500)
    parser.add_argument("--hold-last-ms", type=int, default=2500)
    parser.add_argument("--width", type=int, default=300, help="resize each panel to this width (keeps aspect ratio)")
    parser.add_argument("--gap", type=int, default=16, help="pixel gap between columns")
    parser.add_argument("--label-height", type=int, default=26)
    args = parser.parse_args()

    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match the number of run_dirs")

    per_run_frames = []
    for run_dir in args.run_dirs:
        frames = sorted(Path(run_dir, "samples").glob("epoch_*.png"))
        if not frames:
            raise RuntimeError(f"No sample frames found under {run_dir}/samples")
        per_run_frames.append(frames)

    n_frames = min(len(f) for f in per_run_frames)  # align to the shortest run

    font = ImageFont.load_default()
    composed_frames = []
    for i in range(n_frames):
        panels = []
        for frames in per_run_frames:
            img = Image.open(frames[i]).convert("RGB")
            img = img.resize((args.width, round(img.height * args.width / img.width)), Image.LANCZOS)
            panels.append(img)

        panel_w, panel_h = panels[0].size
        canvas = Image.new(
            "RGB", (panel_w * len(panels) + args.gap * (len(panels) - 1), panel_h + args.label_height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for col, (panel, label) in enumerate(zip(panels, labels)):
            x = col * (panel_w + args.gap)
            canvas.paste(panel, (x, args.label_height))
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (panel_w - text_w) // 2, 6), label, fill="black", font=font)
        composed_frames.append(canvas)

    durations = (
        [args.hold_first_ms] + [args.duration_ms] * (len(composed_frames) - 2) + [args.hold_last_ms]
        if len(composed_frames) > 1
        else [args.hold_last_ms]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed_frames[0].save(
        output_path,
        save_all=True,
        append_images=composed_frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved {len(composed_frames)}-frame GIF ({output_path.stat().st_size / 1024:.0f} KB) to {output_path}")
