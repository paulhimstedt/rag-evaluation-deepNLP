#!/bin/bash
#
# Upload prepared datasets to Modal volume
#
# This script uploads all locally-prepared evaluation datasets to the Modal
# persistent volume so they can be used for RAG evaluation.
#
# Prerequisites:
#   1. Run prepare_datasets_local.py first
#   2. Have Modal CLI installed and authenticated
#
# Usage:
#   bash upload_datasets_to_modal.sh [--dry-run]

set -e  # Exit on error

VOLUME_NAME="rag-data"
LOCAL_DIR="eval_datasets"
REMOTE_DIR="eval_datasets"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "Upload Datasets to Modal Volume"
echo "========================================================================"
echo "Volume: $VOLUME_NAME"
echo "Local:  $LOCAL_DIR/"
echo "Remote: $REMOTE_DIR/"
echo "========================================================================"
echo ""

# Check if local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "❌ Error: Directory '$LOCAL_DIR' not found!"
    echo "   Please run: python prepare_datasets_local.py"
    exit 1
fi

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Error: Modal CLI not found!"
    echo "   Please install: pip install modal"
    exit 1
fi

# Count dataset files
SOURCE_FILES=$(find "$LOCAL_DIR" -name "*.source" -type f | wc -l | tr -d ' ')
TARGET_FILES=$(find "$LOCAL_DIR" -name "*.target" -type f | wc -l | tr -d ' ')

if [ "$SOURCE_FILES" -eq 0 ]; then
    echo "❌ Error: No .source files found in $LOCAL_DIR/"
    echo "   Please run: python prepare_datasets_local.py"
    exit 1
fi

echo "Found $SOURCE_FILES .source files and $TARGET_FILES .target files"
echo ""

# Dry run check
if [ "$1" = "--dry-run" ]; then
    echo "🔍 DRY RUN MODE - No files will be uploaded"
    echo ""
    echo "Would upload:"
    find "$LOCAL_DIR" -type f \( -name "*.source" -o -name "*.target" -o -name "*.csv" \) | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "  $file ($size)"
    done
    echo ""
    echo "To actually upload, run without --dry-run:"
    echo "  bash upload_datasets_to_modal.sh"
    exit 0
fi

# Upload each dataset file
echo "Uploading dataset files..."
echo ""

UPLOAD_COUNT=0
FAILED_COUNT=0

find "$LOCAL_DIR" -type f \( -name "*.source" -o -name "*.target" -o -name "*.csv" \) | sort | while read file; do
    filename=$(basename "$file")
    remote_path="$REMOTE_DIR/$filename"
    size=$(du -h "$file" | cut -f1)
    
    echo -e "${BLUE}Uploading${NC} $filename ($size)..."
    
    if modal volume put "$VOLUME_NAME" "$file" "$remote_path"; then
        echo -e "${GREEN}✓${NC} Uploaded $filename"
        UPLOAD_COUNT=$((UPLOAD_COUNT + 1))
    else
        echo -e "${YELLOW}✗${NC} Failed to upload $filename"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    echo ""
done

echo "========================================================================"
echo "Upload Complete!"
echo "========================================================================"
echo ""

# Verify uploads
echo "Verifying uploaded files..."
echo ""
modal volume ls "$VOLUME_NAME" "$REMOTE_DIR" || echo "Note: Could not list remote files"
echo ""

echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo "1. Verify datasets on Modal volume:"
echo "   modal volume ls $VOLUME_NAME $REMOTE_DIR"
echo ""
echo "2. Run test evaluation (5 samples):"
echo "   modal run modal_rag_eval.py --test-mode"
echo ""
echo "3. Run full evaluation:"
echo "   modal run modal_rag_eval.py"
echo ""
echo "To download results later:"
echo "   modal volume get $VOLUME_NAME results/evaluation_results.json ./"
echo ""
echo "📚 Documentation: See docs/DATASET_PREPARATION_GUIDE.md"
echo "========================================================================"
