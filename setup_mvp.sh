#!/bin/bash
# MVP Setup Script for HRM-RAG System
# Run this to set up your development environment

set -e  # Exit on error

echo "=========================================="
echo "HRM-RAG MVP Setup"
echo "=========================================="

# Check Python version
echo "\n[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Install dependencies
echo "\n[2/6] Installing dependencies..."
echo "  This may take a few minutes..."
pip install -r requirements-mvp.txt -q

# Create directory structure
echo "\n[3/6] Creating directory structure..."
mkdir -p outputs
mkdir -p data
mkdir -p logs/audit
mkdir -p rag_data
mkdir -p models/checkpoints
mkdir -p security
mkdir -p tests
mkdir -p .claude/skills

echo "  ✓ Created directories"

# Generate security keys
echo "\n[4/6] Generating security keys..."

# Generate JWT secret
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "JWT_SECRET_KEY='$JWT_SECRET'" >> .env

# Generate encryption key
ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
echo "ENCRYPTION_KEY='$ENCRYPTION_KEY'" >> .env

echo "  ✓ Generated keys (saved to .env)"
echo "  ⚠️  CRITICAL: Keep .env file secure!"

# Test imports
echo "\n[5/6] Testing imports..."
python3 -c "
import torch
import numpy as np
from security.auth import AuthManager
from security.encryption import EncryptionManager
from security.audit_log import AuditLogger
print('  ✓ All core modules imported successfully')
"

# Create test user
echo "\n[6/6] Creating test admin user..."
python3 << 'EOF'
from security.auth import AuthManager
import os

# Load secret from .env
secret = os.getenv('JWT_SECRET_KEY', 'test-secret-key')

auth = AuthManager(secret_key=secret)
user, api_key = auth.create_user(
    email="admin@test.com",
    organization="Test Organization",
    role="admin"
)

# Save to file
auth.save_to_file('auth_data.json')

print(f"\n  ✓ Created admin user")
print(f"  Email: admin@test.com")
print(f"  API Key: {api_key}")
print(f"\n  ⚠️  SAVE THIS API KEY! It won't be shown again.")
print(f"  ⚠️  Add to your .env file:")
print(f"  TEST_API_KEY='{api_key}'")
EOF

# Summary
echo "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "✓ Dependencies installed"
echo "✓ Directory structure created"
echo "✓ Security keys generated"
echo "✓ Test admin user created"
echo ""
echo "Next steps:"
echo "  1. Review .env file (contains secrets)"
echo "  2. Add .env to .gitignore"
echo "  3. Read MVP_ROADMAP.md"
echo "  4. Test with: python security/auth.py"
echo ""
echo "To train a quick test model:"
echo "  ./run_quick_test.sh"
echo ""
echo "=========================================="
