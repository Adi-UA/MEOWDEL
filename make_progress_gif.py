import argparse
from pathlib import Path

from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="e.g. outputs/20260708_104330")
    parser.add_argument("--output", default="assets/training_progress.gif")
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--hold-last-ms", type=int, default=1500)
    parser.add_argument("--width", type=int, default=300, help="resize frames to this width (keeps aspect ratio)")
    args = parser.parse_args()

    frames = sorted(Path(args.run_dir, "samples").glob("epoch_*.png"))
    if not frames:
        raise RuntimeError(f"No sample frames found under {args.run_dir}/samples")

    images = [Image.open(f).convert("RGB") for f in frames]
    if args.width:
        images = [
            img.resize(
                (args.width, round(img.height * args.width / img.width)),
                Image.LANCZOS,
            )
            for img in images
        ]
    durations = [args.duration_ms] * (len(images) - 1) + [args.hold_last_ms]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved {len(images)}-frame GIF ({output_path.stat().st_size / 1024:.0f} KB) to {output_path}")
