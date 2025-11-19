"""
Security module for HRM-RAG system
Provides authentication, encryption, and audit logging for HIPAA/SOC2 compliance
"""

from .auth import AuthManager, User, require_auth, require_permission
from .encryption import EncryptionManager
from .audit_log import AuditLogger

__all__ = [
    'AuthManager',
    'User',
    'require_auth',
    'require_permission',
    'EncryptionManager',
    'AuditLogger',
]
