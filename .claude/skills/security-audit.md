# Security Audit

Run comprehensive security checks on the codebase.

## Purpose
- Check for security vulnerabilities
- Verify encryption implementation
- Check authentication
- Find hardcoded secrets
- Validate audit logging

## Usage

```bash
/security-audit
```

## Commands

```bash
echo "=================================="
echo "Security Audit"
echo "=================================="
echo ""

ISSUES=0
WARNINGS=0

# Check 1: Hardcoded secrets
echo "[1/8] Checking for hardcoded secrets..."

if grep -r "password\s*=\s*['\"][^<]" --include="*.py" . 2>/dev/null | grep -v "test" | grep -v ".pyc"; then
    echo "  ✗ Found potential hardcoded passwords"
    ((ISSUES++))
else
    echo "  ✓ No hardcoded passwords"
fi

if grep -r "api_key\s*=\s*['\"][^<]" --include="*.py" . 2>/dev/null | grep -v "test" | grep -v ".pyc" | grep -v "example"; then
    echo "  ✗ Found potential hardcoded API keys"
    ((ISSUES++))
else
    echo "  ✓ No hardcoded API keys"
fi

if grep -r "secret\s*=\s*['\"][^<]" --include="*.py" . 2>/dev/null | grep -v "test" | grep -v ".pyc" | grep -v "your-secret"; then
    echo "  ✗ Found potential hardcoded secrets"
    ((ISSUES++))
else
    echo "  ✓ No hardcoded secrets"
fi

# Check 2: Encryption implementation
echo ""
echo "[2/8] Checking encryption..."

if [ -f "security/encryption.py" ]; then
    if grep -q "Fernet\|AES" security/encryption.py; then
        echo "  ✓ Encryption implementation found"
    else
        echo "  ✗ Encryption module exists but no crypto found"
        ((ISSUES++))
    fi
else
    echo "  ✗ No encryption module found"
    ((ISSUES++))
fi

# Check 3: Authentication
echo ""
echo "[3/8] Checking authentication..."

if [ -f "security/auth.py" ]; then
    if grep -q "jwt\|api_key" security/auth.py; then
        echo "  ✓ Authentication implementation found"
    else
        echo "  ✗ Auth module exists but incomplete"
        ((ISSUES++))
    fi
else
    echo "  ✗ No authentication module"
    ((ISSUES++))
fi

# Check 4: Audit logging
echo ""
echo "[4/8] Checking audit logging..."

if [ -f "security/audit_log.py" ]; then
    echo "  ✓ Audit logging module exists"
else
    echo "  ✗ No audit logging"
    ((ISSUES++))
fi

# Check 5: Environment variables
echo ""
echo "[5/8] Checking environment configuration..."

if [ -f ".env" ]; then
    if grep -q "JWT_SECRET_KEY\|ENCRYPTION_KEY" .env; then
        echo "  ✓ .env file configured"
    else
        echo "  ⚠ .env exists but missing keys"
        ((WARNINGS++))
    fi
else
    echo "  ⚠ No .env file found"
    ((WARNINGS++))
fi

if grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo "  ✓ .env in .gitignore"
else
    echo "  ✗ .env NOT in .gitignore - SECURITY RISK!"
    ((ISSUES++))
fi

# Check 6: Dependencies
echo ""
echo "[6/8] Checking dependencies for vulnerabilities..."

if command -v safety &> /dev/null; then
    if safety check --json > /dev/null 2>&1; then
        echo "  ✓ No known vulnerabilities in dependencies"
    else
        echo "  ⚠ Vulnerabilities found in dependencies"
        echo "    Run: safety check --full-report"
        ((WARNINGS++))
    fi
else
    echo "  ⚠ 'safety' not installed - cannot check vulnerabilities"
    echo "    Install: pip install safety"
    ((WARNINGS++))
fi

# Check 7: HTTPS/TLS
echo ""
echo "[7/8] Checking HTTPS/TLS configuration..."

if grep -r "ssl_context\|https\|tls" --include="*.py" . 2>/dev/null | grep -v ".pyc" > /dev/null; then
    echo "  ✓ HTTPS/TLS configuration found"
else
    echo "  ⚠ No HTTPS/TLS configuration"
    echo "    Required for production deployment"
    ((WARNINGS++))
fi

# Check 8: SQL Injection protection
echo ""
echo "[8/8] Checking SQL injection protection..."

if grep -r "cursor\.execute.*%\|cursor\.execute.*format" --include="*.py" . 2>/dev/null | grep -v ".pyc"; then
    echo "  ✗ Potential SQL injection vulnerabilities"
    echo "    Use parameterized queries instead"
    ((ISSUES++))
else
    echo "  ✓ No obvious SQL injection risks"
fi

# Summary
echo ""
echo "=================================="
echo "Security Audit Summary"
echo "=================================="
echo "Critical Issues: $ISSUES"
echo "Warnings: $WARNINGS"
echo ""

if [ $ISSUES -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ Security audit passed!"
    echo ""
    exit 0
elif [ $ISSUES -eq 0 ]; then
    echo "⚠ Security audit passed with warnings"
    echo "  Address warnings before production deployment"
    echo ""
    exit 0
else
    echo "✗ Security audit failed"
    echo "  Fix critical issues before proceeding"
    echo ""
    exit 1
fi
```

## Common Issues & Fixes

### Hardcoded Secrets
```bash
# Bad
password = "mysecret123"

# Good
password = os.environ.get('DB_PASSWORD')
```

### .env Not in .gitignore
```bash
echo ".env" >> .gitignore
```

### Missing Encryption
```bash
# Install security modules
./setup_mvp.sh
```

## Install Security Tools

```bash
pip install safety bandit
```

## Advanced Audit

For deeper analysis:
```bash
# Check with bandit
bandit -r . -f json -o security_report.json

# Check dependencies
safety check --full-report

# Check for secrets in git history
git secrets --scan-history
```

## Success Criteria

- [x] No hardcoded secrets
- [x] Encryption implemented
- [x] Authentication implemented
- [x] Audit logging present
- [x] .env in .gitignore
- [x] No known vulnerabilities
- [x] HTTPS configured (for production)
