#!/usr/bin/env bash
# Assemble a static demo site for CityMind: dashboard pages + precomputed
# /api data (served as extension-less files) + overlay/frame images.
set -e
rm -rf dist
mkdir -p dist/api dist/overlays dist/frames
# index_final.html is the production page (served at / by the original server.py)
cp dashboard/index_final.html dist/index.html
cp dashboard/index_final.html dist/index_final.html
cp dashboard/index.html dist/classic.html
cp dashboard/platform.html dashboard/demo.html dist/ 2>/dev/null || true
cp dashboard/*.webp dashboard/*.png dist/ 2>/dev/null || true
SRC=output/demo_rich
cp "$SRC/twin.json"                dist/api/twin
cp "$SRC/defects.json"             dist/api/defects
cp "$SRC/agents.json"              dist/api/agents
cp "$SRC/performance.json"         dist/api/performance
cp "$SRC/scan_history.json"        dist/api/scan-history
cp "$SRC/structural_elements.json" dist/api/structural-elements
cp "$SRC/pipeline.json"            dist/api/pipeline
cp "$SRC/demo_data_complete.json"  dist/api/complete
cp "$SRC/defect_overlays/"*.png dist/overlays/ 2>/dev/null || true
cp "$SRC/frames/"*.jpg dist/frames/ 2>/dev/null || true
node -e "const fs=require('fs');const d='$SRC/defect_overlays';const ov=fs.existsSync(d)?fs.readdirSync(d).filter(f=>f.endsWith('.png')).sort():[];fs.writeFileSync('dist/api/overlays',JSON.stringify(ov));const fd='$SRC/frames';const fr=fs.existsSync(fd)?fs.readdirSync(fd).filter(f=>f.endsWith('.jpg')).sort():[];fs.writeFileSync('dist/api/frames',JSON.stringify(fr));"
echo "Assembled dist/ ($(du -sh dist | cut -f1))"
