#!/bin/bash
# Mobile Gaze Pipeline — launcher
#
# Usage:
#   bash run.sh <command> --session <path> [--weight <path>] [--dataset <name>] [--stride N] [--skip-ms N] [--verbose]
#
# Commands:
#   viz   Visualizer                  — verify timestamp alignment before inference
#   val   Calibration + Validation    — full pipeline, outputs residual + scatter plots
#
# Options:
#   --session   Path to session folder (required)
#   --weight    Path to model weights  (default: weights/resnet34.pt)
#   --dataset   Dataset config: gaze360 | mpiigaze  (default: gaze360)
#   --stride    Process every Nth frame, 1=all frames (default: 1)
#   --skip-ms       ms to skip after each point_start       (default: 1000)
#   --end-trim-ms   ms to trim before each point_end        (default: 500)
#   --verbose   Enable DEBUG logging
#
# Examples:
#   bash run.sh viz --session GazeData/session_20260504_210116_THH0FFPA
#   bash run.sh val --session GazeData/session_20260504_210116_THH0FFPA
#   bash run.sh val --session GazeData/session_xxx --weight weights/resnet34_mpiigaze.pt --dataset mpiigaze
#   bash run.sh val --session GazeData/session_xxx --stride 3   # faster, every 3rd frame

set -e

PYTHON=/opt/anaconda3/envs/DeepLearning/bin/python

# ── Defaults ──────────────────────────────────────────────────────────────────
CMD=""
SESSION=""
WEIGHT="weights/resnet34.pt"
DATASET="gaze360"
STRIDE=1
SKIP_MS=1000
END_TRIM_MS=1000
VERBOSE=""

# ── Parse arguments ───────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    grep "^#" "$0" | sed 's/^# *//' | head -30
    exit 0
fi

# First positional argument is the command
case $1 in
    viz|val) CMD=$1; shift ;;
    *)
        echo "ERROR: first argument must be a command: viz | val"
        echo "Run 'bash run.sh' with no arguments for usage."
        exit 1
        ;;
esac

while [[ $# -gt 0 ]]; do
    case $1 in
        --session)   SESSION=$2;  shift 2 ;;
        --weight)    WEIGHT=$2;   shift 2 ;;
        --dataset)   DATASET=$2;  shift 2 ;;
        --stride)    STRIDE=$2;   shift 2 ;;
        --skip-ms)      SKIP_MS=$2;      shift 2 ;;
        --end-trim-ms)  END_TRIM_MS=$2;  shift 2 ;;
        --verbose)      VERBOSE="--verbose"; shift ;;
        *)
            echo "ERROR: unknown option '$1'"
            echo "Run 'bash run.sh' with no arguments for usage."
            exit 1
            ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$SESSION" ]]; then
    echo "ERROR: --session is required."
    exit 1
fi

if [[ ! -d "$SESSION" ]]; then
    echo "ERROR: session folder not found: $SESSION"
    exit 1
fi

if [[ "$CMD" != "viz" && ! -f "$WEIGHT" ]]; then
    echo "ERROR: weight file not found: $WEIGHT"
    exit 1
fi

if [[ "$DATASET" != "gaze360" && "$DATASET" != "mpiigaze" ]]; then
    echo "ERROR: --dataset must be gaze360 or mpiigaze, got: $DATASET"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────────────────"
echo "  command : $CMD"
echo "  session : $SESSION"
[[ "$CMD" != "viz" ]] && echo "  weight  : $WEIGHT"
[[ "$CMD" != "viz" ]] && echo "  dataset : $DATASET"
[[ "$CMD" != "viz" ]] && echo "  stride  : $STRIDE"
echo "  skip-ms     : $SKIP_MS"
echo "  end-trim-ms : $END_TRIM_MS"
echo "────────────────────────────────────────────────"

case $CMD in
    viz)
        $PYTHON pipeline/visualizer.py \
            --session      "$SESSION" \
            --skip-ms      "$SKIP_MS" \
            --end-trim-ms  "$END_TRIM_MS" \
            $VERBOSE
        ;;
    val)
        $PYTHON pipeline/run_validation.py \
            --session      "$SESSION" \
            --weight       "$WEIGHT" \
            --dataset      "$DATASET" \
            --stride       "$STRIDE" \
            --skip-ms      "$SKIP_MS" \
            --end-trim-ms  "$END_TRIM_MS" \
            $VERBOSE
        ;;
esac
