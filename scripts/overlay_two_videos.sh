#!/bin/bash
# Usage: ./overlay_videos.sh background.mp4 overlay.mp4 output.mp4 [opacity] [x_offset] [y_offset]
# Example: ./overlay_videos.sh bg.mp4 overlay.mp4 out.mp4 0.5 100 50

# --- Args ---
BACKGROUND=$1
OVERLAY=$2
OUTPUT=$3
OPACITY=${4:-0.5}       # Default opacity = 0.5
X_OFFSET=${5:-0}        # Default position x = 0
Y_OFFSET=${6:-0}        # Default position y = 0

# --- Validation ---
if [[ -z "$BACKGROUND" || -z "$OVERLAY" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 background.mp4 overlay.mp4 output.mp4 [opacity] [x_offset] [y_offset]"
    exit 1
fi

# --- Run FFmpeg ---
ffmpeg -i "$BACKGROUND" -i "$OVERLAY" \
-filter_complex "[1:v]format=rgba,colorchannelmixer=aa=$OPACITY[ov];[0:v][ov]overlay=$X_OFFSET:$Y_OFFSET" \
-shortest -c:a copy "$OUTPUT"
