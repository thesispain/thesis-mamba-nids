#!/bin/bash
# ============================================================
# auto_commit.sh — Commit & Push to GitHub every 4 hours
# Usage: nohup bash auto_commit.sh &
# ============================================================

REPO_DIR="/home/T2510596/Downloads/totally fresh/thesis_final"
INTERVAL=14400  # 4 hours in seconds
REMOTE="origin"
BRANCH="main"

echo "🚀 Auto-commit started at $(date)"
echo "   Repo: $REPO_DIR"
echo "   Interval: ${INTERVAL}s (4 hours)"
echo "   Press Ctrl+C to stop"

while true; do
    cd "$REPO_DIR" || exit 1

    # Stage all changes
    git add .

    # Check if there are changes to commit
    if git diff --cached --quiet; then
        echo "⏭️  $(date): No changes to commit. Sleeping..."
    else
        # Commit with timestamp
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
        git commit -m "Auto-commit: $TIMESTAMP"
        
        # Push
        echo "📤 $(date): Pushing to $REMOTE/$BRANCH..."
        git push $REMOTE $BRANCH 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ $(date): Push successful!"
        else
            echo "❌ $(date): Push failed. Will retry next cycle."
        fi
    fi

    echo "💤 Sleeping for ${INTERVAL}s (4 hours)..."
    sleep $INTERVAL
done
