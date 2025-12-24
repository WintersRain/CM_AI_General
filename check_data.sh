#!/bin/bash
# Quick commands for CM AI development

# Check recorded data
echo "=== Recorded Sessions ==="
ls -la data/

echo ""
echo "=== Sample Actions ==="
head -50 data/*/actions.json 2>/dev/null || echo "No recordings found yet"

echo ""
echo "=== Frame Count ==="
ls data/*/*.npy 2>/dev/null | wc -l
