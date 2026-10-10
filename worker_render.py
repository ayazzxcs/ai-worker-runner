import sys
import os
import subprocess
import glob
from pathlib import Path

def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    vf_filter = sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        duration = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0.0
    except ValueError:
        duration = 0.0
    try:
        seek_offset = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 0.0
    except ValueError:
        seek_offset = 0.0

    # Locate input video cut
    input_video = None
    for candidate in ("input_cut.mp4", "cut.mp4"):
        if os.path.exists(candidate):
            input_video = candidate
            break
    if not input_video:
        mp4_files = glob.glob("*.mp4")
        if mp4_files:
            input_video = mp4_files[0]

    if not input_video or not os.path.exists(input_video):
        print(f"ERROR: No input MP4 found in working directory!", file=sys.stderr)
        sys.exit(1)

    print(f"[Worker Render] Processing {input_video} for job {job_id}...")
    print(f"  Duration: {duration}s, Seek offset: {seek_offset}s")
    print(f"  Filter: {vf_filter}")

    # Build filtergraph
    filters = []
    if vf_filter and vf_filter.strip():
        filters.append(vf_filter.strip())

    subtitles_file = None
    for sub_candidate in ("subtitles.ass", "subs.ass"):
        if os.path.exists(sub_candidate) and os.path.getsize(sub_candidate) > 10:
            subtitles_file = sub_candidate
            break

    if subtitles_file:
        print(f"  Burning subtitles from {subtitles_file}...")
        filters.append(f"ass={subtitles_file}")

    cmd = ["ffmpeg", "-y"]
    if seek_offset > 0.001:
        cmd.extend(["-ss", str(round(seek_offset, 3))])
    cmd.extend(["-i", input_video])
    if duration > 0.001:
        cmd.extend(["-t", str(round(duration, 3))])

    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "rendered.mp4"
    ])

    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"FFmpeg failed with exit code {res.returncode}:\n{res.stderr[-1000:]}", file=sys.stderr)
        sys.exit(res.returncode)

    out_size_mb = os.path.getsize("rendered.mp4") / (1024 * 1024)
    print(f"SUCCESS: Rendered {out_size_mb:.2f} MB to rendered.mp4")

if __name__ == "__main__":
    main()
