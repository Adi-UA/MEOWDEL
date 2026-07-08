import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs", nargs="+", help="one or more run dirs, e.g. outputs/20260708_104330 outputs/20260708_121740"
    )
    parser.add_argument("--labels", nargs="+", help="one label per run_dir; defaults to each run dir's folder name")
    parser.add_argument("--output", default="assets/comparison.png")
    parser.add_argument("--width", type=int, default=220, help="resize each panel to this width (keeps aspect ratio)")
    parser.add_argument("--gap", type=int, default=16, help="pixel gap between columns/rows")
    parser.add_argument("--label-height", type=int, default=26)
    args = parser.parse_args()

    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match the number of run_dirs")

    columns = []  # each column: (label, first_frame_img, last_frame_img, first_epoch, last_epoch)
    for run_dir, label in zip(args.run_dirs, labels):
        frames = sorted(Path(run_dir, "samples").glob("epoch_*.png"))
        if len(frames) < 2:
            raise RuntimeError(f"Need at least 2 sample frames under {run_dir}/samples")
        epoch_re = re.compile(r"epoch_(\d+)")
        first_epoch = int(epoch_re.search(frames[0].stem).group(1))
        last_epoch = int(epoch_re.search(frames[-1].stem).group(1))
        panels = []
        for frame in (frames[0], frames[-1]):
            img = Image.open(frame).convert("RGB")
            img = img.resize((args.width, round(img.height * args.width / img.width)), Image.LANCZOS)
            panels.append(img)
        columns.append((label, panels[0], panels[1], first_epoch, last_epoch))

    panel_w, panel_h = columns[0][1].size
    n_cols = len(columns)
    row_label_w = 90
    canvas_w = row_label_w + panel_w * n_cols + args.gap * (n_cols - 1)
    canvas_h = args.label_height + panel_h * 2 + args.gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col, (label, first_img, last_img, first_epoch, last_epoch) in enumerate(columns):
        x = row_label_w + col * (panel_w + args.gap)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (panel_w - text_w) // 2, 6), label, fill="black", font=font)
        canvas.paste(first_img, (x, args.label_height))
        canvas.paste(last_img, (x, args.label_height + panel_h + args.gap))

    for row, epoch_label in enumerate(("epoch 0", "final")):
        y = args.label_height + row * (panel_h + args.gap) + panel_h // 2
        draw.text((6, y), epoch_label, fill="black", font=font)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved comparison ({output_path.stat().st_size / 1024:.0f} KB) to {output_path}")
