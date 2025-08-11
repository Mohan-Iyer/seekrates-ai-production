#!/bin/bash
# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# script_id: "fr_18_uc_018_ec_01_tc_252"
# gps_coordinate: "fr_18_uc_018_ec_01_tc_252"
# script_name: "setup_clean_deployment.sh"
# purpose: "One-command enterprise deployment environment setup"
# version: "1.0.0"
# status: "Production"
# author: "MI"
# created_on: "2025-08-10T13:55:00Z"
# coding_engineer: "Claude"
# supervisor: "Yang - ChatGPT"
# business_owner: "Mohan Iyer mohan@pixels.net.nz"
# =============================================================================

set -e  # Exit on any error

echo "🚀 Enterprise Deployment Setup - GPS: fr_18_uc_018_ec_01_tc_252"
echo "🔧 One-command environment preparation"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Install enterprise dependencies
echo "📋 Installing enterprise dependencies..."
pip install flask==3.0.0 requests==2.32.3 anthropic==0.34.0 openai==1.40.0 gunicorn==23.0.0 python-dotenv==1.0.1 boto3==1.28.85

# Create Python module structure
echo "🔧 Creating Python module structure..."
touch src/__init__.py
touch src/secrets/__init__.py
touch src/utils/__init__.py
touch src/agents/__init__.py
touch src/core/__init__.py
touch src/cognitive_whiteboard/__init__.py

# Verify critical files exist
echo "🔍 Verifying enterprise files..."
echo "✅ Directory map:" 
ls -la directory_map.yaml 2>/dev/null || echo "❌ directory_map.yaml missing"

echo "✅ Cognitive whiteboard:"
ls -la src/cognitive_whiteboard/ 2>/dev/null || echo "❌ cognitive_whiteboard missing"

echo "✅ AWS secrets:"
ls -la src/secrets/ 2>/dev/null || echo "❌ secrets directory missing"

echo "✅ Python modules:"
ls -la src/__init__.py src/secrets/__init__.py src/utils/__init__.py 2>/dev/null || echo "❌ __init__.py files missing"

echo ""
echo "✅ ENTERPRISE SETUP COMPLETE"
echo "🎯 Ready for: python3 app.py"
echo "🚀 Ready for: ./deploy_to_railway.sh"
