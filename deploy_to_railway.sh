#!/bin/bash
# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT  
# =============================================================================
# project_name: "decision_referee"
# script_id: "fr_11_uc_091_ec_01_tc_250"
# gps_coordinate: "fr_11_uc_091_ec_01_tc_250"
# script_name: "deploy_to_railway.sh"
# purpose: "Railway deployment via GitHub with professional exclusions"
# version: "1.0.0" 
# status: "Production"
# author: "MI"
# created_on: "2025-08-10T12:40:00Z"
# coding_engineer: "Claude"
# supervisor: "Yang - ChatGPT"
# business_owner: "Mohan Iyer mohan@pixels.net.nz"
# =============================================================================

# Strict bash settings for safety
set -Eeuo pipefail
IFS=$'\n\t'

echo "🚀 Railway Deployment - GPS: fr_11_uc_091_ec_01_tc_250"
echo "🧬 Professional deployment excluding venv/ (Railway creates own)"
echo ""

# Helper function to check required commands
require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ Required command '$cmd' not found"
        exit 1
    fi
}

# Check required commands exist
require_cmd git
require_cmd date

# Verify essential files exist and are non-empty
if [ ! -f "app.py" ] || [ ! -s "app.py" ]; then
    echo "❌ app.py missing or empty"
    exit 1
fi

if [ ! -f "requirements.txt" ] || [ ! -s "requirements.txt" ]; then
    echo "❌ requirements.txt missing or empty"
    exit 1
fi

echo "✅ Essential files validated"

# Verify git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git branch -M main
    
    # Require origin remote to be set
    if ! git remote get-url origin &> /dev/null; then
        echo "❌ Git initialized but no origin remote configured"
        echo "🔧 Run: git remote add origin <your-github-repo-url>"
        echo "   Example: git remote add origin https://github.com/username/repo.git"
        exit 1
    fi
fi

# Verify origin remote exists
if ! git remote get-url origin &> /dev/null; then
    echo "❌ No git origin remote configured"
    echo "🔧 Run: git remote add origin <your-github-repo-url>"
    exit 1
fi

echo "✅ Git repository and origin remote validated"

# Stage all files
git add -A

# Guard: Check for forbidden paths in staged files
forbidden_patterns="venv/|__pycache__/|\.env|\.key|secrets/.*\.encrypted|docs/|templates/|.*hydration.*|research/|\.broken"

if git ls-files -s | grep -E "$forbidden_patterns" > /dev/null; then
    echo "❌ Forbidden files detected in staging area:"
    git ls-files -s | grep -E "$forbidden_patterns"
    echo "🔧 Check your .gitignore and remove these files from staging"
    exit 1
fi

echo "✅ No forbidden files in staging area"

# Show git status (non-interactive)
echo "📦 Current git status:"
git status --porcelain

echo ""
echo "📦 Files for Railway deployment:"
git ls-files | head -20
echo "... (essential files only, venv/ excluded)"
echo ""

# Commit with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "🚀 Railway deployment - $TIMESTAMP" || echo "ℹ️  No changes to commit"

# Push to origin main
echo "🔄 Pushing to origin main..."
git push origin main

echo ""
echo "✅ RAILWAY DEPLOYMENT TRIGGERED"
echo "🌐 Railway will create fresh environment from requirements.txt"
echo "📊 Repository: $(git remote get-url origin)"
echo "🎯 Branch: main"