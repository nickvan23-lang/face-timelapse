#!/usr/bin/env python3
import argparse
import sys
from typing import Tuple
from facelapse import discover_images, detect_all_faces, smooth_eye_coords, write_video

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a stabilized face timelapse.")
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", default="timelapse.mp4", help="Output file")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--fps", type=float, default=24.0, help="Frames per second")
    parser.add_argument("--res", type=str, default="1920x1080", help="Resolution WxH")
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor")
    parser.add_argument("--alpha", type=float, default=0.25, help="EMA smoothing factor")
    parser.add_argument("--span", type=float, default=0.35, help="Eye span fraction")
    parser.add_argument("--no-blur", action="store_true", help="Use black bars instead of blurred background")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of worker processes")
    return parser

def parse_resolution(spec: str) -> Tuple[int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError("Resolution must be WxH (e.g. 1920x1080)")
    return int(parts[0]), int(parts[1])

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        w, h = parse_resolution(args.res)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovering images in {args.input}...")
    try:
        paths = discover_images(args.input, recursive=args.recursive)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} images. Detecting faces...")
    def print_progress(phase: str, current: int, total: int) -> None:
        print(f"\r{phase.capitalize()}: {current}/{total}", end="")

    detections = detect_all_faces(paths, num_workers=args.workers, progress_callback=print_progress)
    print("\nSmoothing coordinates...")
    smoothed = smooth_eye_coords(detections, alpha=args.alpha)

    print("Writing video...")
    write_video(
        paths, smoothed, args.output,
        fps=args.fps, dst_w=w, dst_h=h,
        zoom=args.zoom, eye_span_fraction=args.span,
        blur_bg=not args.no_blur,
        progress_callback=print_progress
    )
    print(f"\nDone! Saved to {args.output}")

if __name__ == "__main__":
    main()
