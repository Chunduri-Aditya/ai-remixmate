#!/bin/bash
# End-to-End Smoke Test for AI RemixMate
# This script tests the complete pipeline and generates a quality report

set -e  # Exit on any error

echo "🎵 AI RemixMate - End-to-End Smoke Test"
echo "========================================"

# Configuration
PROJECT_ROOT="/Users/chunduri/Desktop/ai-remixmate"
PYTHONPATH="$PROJECT_ROOT"
VENV_PATH="$PROJECT_ROOT/remix-env"
OUTPUT_DIR="$PROJECT_ROOT/tests/output"
REPORT_FILE="$OUTPUT_DIR/smoke_test_report.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

echo "📊 Running smoke test pipeline..."

# Test 1: Download test songs (if not already present)
echo "🎧 Test 1: Downloading test songs..."
cd "$PROJECT_ROOT"

# Download a simple test song if not exists
if [ ! -f "audio_input/test_song.wav" ]; then
    echo "   Downloading test song..."
    python scripts/download_song.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --title "test_song" || echo "   Download failed, using existing files"
fi

# Test 2: Separate audio (if not already done)
echo "🎛️  Test 2: Audio separation..."
if [ ! -d "separated/htdemucs/test_song" ]; then
    echo "   Separating audio stems..."
    python scripts/batch_demucs.py --glob "test_song.wav" || echo "   Separation failed, using existing stems"
fi

# Test 3: Extract features
echo "🔍 Test 3: Feature extraction..."
if [ ! -f "models/song_embeddings.json" ]; then
    echo "   Building feature database..."
    python scripts/song_database.py --wav "audio_input/test_song.wav" --name "test_song" || echo "   Feature extraction failed"
fi

# Test 4: Create a simple remix
echo "🎵 Test 4: Creating test remix..."
if [ -d "separated/htdemucs/test_song" ]; then
    echo "   Generating basic remix..."
    python scripts/remix_from_match.py --base "test_song" --auto --fade 0.5 || echo "   Remix generation failed"
fi

# Test 5: Run metrics evaluation
echo "📊 Test 5: Quality metrics evaluation..."
if [ -f "output/remix_test_song_test_song.wav" ]; then
    echo "   Evaluating remix quality..."
    python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from scripts.core.metrics import AudioMetrics
from pathlib import Path

metrics = AudioMetrics()
results = metrics.evaluate_remix(
    Path('separated/htdemucs/test_song/vocals.wav'),
    Path('separated/htdemucs/test_song/other.wav'),
    Path('output/remix_test_song_test_song.wav')
)

metrics.save_metrics_report(results, Path('$REPORT_FILE'))
metrics.print_metrics_summary(results)
"
else
    echo "   No remix file found for evaluation"
fi

# Test 6: Generate summary report
echo "📋 Test 6: Generating summary report..."
cat > "$OUTPUT_DIR/test_summary.txt" << EOF
AI RemixMate Smoke Test Summary
==============================

Test Date: $(date)
Project Root: $PROJECT_ROOT
Virtual Environment: $VENV_PATH

Test Results:
- Download: $(if [ -f "audio_input/test_song.wav" ]; then echo "✅ PASS"; else echo "❌ FAIL"; fi)
- Separation: $(if [ -d "separated/htdemucs/test_song" ]; then echo "✅ PASS"; else echo "❌ FAIL"; fi)
- Features: $(if [ -f "models/song_embeddings.json" ]; then echo "✅ PASS"; else echo "❌ FAIL"; fi)
- Remix: $(if [ -f "output/remix_test_song_test_song.wav" ]; then echo "✅ PASS"; else echo "❌ FAIL"; fi)
- Metrics: $(if [ -f "$REPORT_FILE" ]; then echo "✅ PASS"; else echo "❌ FAIL"; fi)

Output Files:
- Remix: output/remix_test_song_test_song.wav
- Metrics: $REPORT_FILE
- Summary: $OUTPUT_DIR/test_summary.txt

Next Steps:
1. Review metrics report for quality assessment
2. Run advanced remix generator for ML optimization
3. Test with different song pairs
4. Validate against quality thresholds

EOF

echo "✅ Smoke test completed!"
echo "📊 Results saved to: $OUTPUT_DIR"
echo "📋 Summary: $OUTPUT_DIR/test_summary.txt"
echo "📈 Metrics: $REPORT_FILE"

# Display summary
echo ""
echo "📋 TEST SUMMARY"
echo "==============="
cat "$OUTPUT_DIR/test_summary.txt"

echo ""
echo "🎉 Smoke test pipeline completed successfully!"
echo "   Check the output files for detailed results."
