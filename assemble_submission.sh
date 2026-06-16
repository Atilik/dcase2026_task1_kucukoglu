#!/bin/bash
# Assemble DCASE 2026 submission package
# Run AFTER eval_inference.py produces all output CSVs

set -e

TEAM="Kucukoglu_NYU"
TASK="task1"
BASE_DIR="/scratch/mk9649/repos/dcase2026_task1_baseline"
PKG_DIR="${BASE_DIR}/submission_package/${TASK}"

echo "=== Assembling DCASE 2026 Submission Package ==="

# Clean and create directory structure
rm -rf "${BASE_DIR}/submission_package"
for i in 1 2 3 4; do
    mkdir -p "${PKG_DIR}/${TEAM}_${TASK}_${i}"
done

# Copy meta YAML files
for i in 1 2 3 4; do
    src="${BASE_DIR}/${TEAM}_${TASK}_${i}.meta.yaml"
    dst="${PKG_DIR}/${TEAM}_${TASK}_${i}/${TEAM}_${TASK}_${i}.meta.yaml"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "  ✅ Copied ${TEAM}_${TASK}_${i}.meta.yaml"
    else
        echo "  ❌ Missing ${src}"
    fi
done

# Copy output CSVs (produced by eval_inference.py)
for i in 1 2 3 4; do
    src="${BASE_DIR}/${TEAM}_${TASK}_${i}.output.csv"
    dst="${PKG_DIR}/${TEAM}_${TASK}_${i}/${TEAM}_${TASK}_${i}.output.csv"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        lines=$(wc -l < "$src")
        echo "  ✅ Copied ${TEAM}_${TASK}_${i}.output.csv ($lines lines)"
    else
        echo "  ❌ Missing ${src} — run eval_inference.py first!"
    fi
done

# Copy technical report if it exists
REPORT="${BASE_DIR}/${TEAM}_${TASK}_TechnicalReport.pdf"
if [ -f "$REPORT" ]; then
    cp "$REPORT" "${PKG_DIR}/"
    echo "  ✅ Copied technical report"
else
    echo "  ⚠️  No technical report found at ${REPORT}"
    echo "     Create a PDF and place it there before final submission"
fi

# Create zip
cd "${BASE_DIR}/submission_package"
ZIP_FILE="${BASE_DIR}/${TEAM}.zip"
rm -f "$ZIP_FILE"
zip -r "$ZIP_FILE" "${TASK}/"
echo ""
echo "=== Package created: ${ZIP_FILE} ==="
echo ""

# Validate contents
echo "Package contents:"
unzip -l "$ZIP_FILE"
