import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="e.g. outputs/20260708_104330")
    parser.add_argument("--output", default="assets/comparison.png")
    parser.add_argument("--width", type=int, default=300, help="resize each panel to this width (keeps aspect ratio)")
    parser.add_argument("--gap", type=int, default=20, help="pixel gap between panels")
    parser.add_argument("--label-height", type=int, default=30)
    args = parser.parse_args()

    frames = sorted(Path(args.run_dir, "samples").glob("epoch_*.png"))
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 sample frames under {args.run_dir}/samples")

    first_frame, last_frame = frames[0], frames[-1]
    epoch_re = re.compile(r"epoch_(\d+)")
    first_epoch = int(epoch_re.search(first_frame.stem).group(1))
    last_epoch = int(epoch_re.search(last_frame.stem).group(1))

    panels = []
    for frame in (first_frame, last_frame):
        img = Image.open(frame).convert("RGB")
        img = img.resize((args.width, round(img.height * args.width / img.width)), Image.LANCZOS)
        panels.append(img)

    panel_w, panel_h = panels[0].size
    canvas = Image.new("RGB", (panel_w * 2 + args.gap, panel_h + args.label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, (panel, epoch) in enumerate(zip(panels, (first_epoch, last_epoch))):
        x = i * (panel_w + args.gap)
        canvas.paste(panel, (x, args.label_height))
        label = f"Epoch {epoch:03d}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (panel_w - text_w) // 2, 5), label, fill="black", font=font)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved comparison ({output_path.stat().st_size / 1024:.0f} KB) to {output_path}")
