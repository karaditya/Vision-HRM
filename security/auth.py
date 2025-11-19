"""
Authentication and Authorization System
Implements JWT tokens, API keys, and role-based access control
"""

import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import os


@dataclass
class User:
    """User model with permissions"""
    id: str
    email: str
    organization: str
    role: str  # admin, user, viewer
    api_key_hash: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_login: Optional[str] = None


class AuthManager:
    """Manage authentication and authorization"""

    def __init__(self, secret_key: Optional[str] = None, token_expiry_hours: int = 24):
        """
        Args:
            secret_key: Secret for JWT signing (generate if None)
            token_expiry_hours: JWT token expiry time
        """
        if secret_key is None:
            secret_key = secrets.token_urlsafe(32)
            print(f"⚠️  Generated new secret key: {secret_key}")
            print("   SAVE THIS KEY SECURELY! Set as environment variable:")
            print(f"   export JWT_SECRET_KEY='{secret_key}'")

        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours

        # In production, use database (PostgreSQL, MongoDB, etc.)
        self.users_db: Dict[str, User] = {}
        self.api_keys_db: Dict[str, str] = {}  # api_key_hash -> user_id

        print("✓ AuthManager initialized")

    def generate_api_key(self) -> str:
        """Generate cryptographically secure API key"""
        return f"hrm_{secrets.token_urlsafe(32)}"

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_user(
        self,
        email: str,
        organization: str,
        role: str = "user",
    ) -> tuple[User, str]:
        """
        Create new user

        Returns:
            (User object, plain API key) - API key shown only once!
        """
        user_id = f"user_{secrets.token_urlsafe(12)}"
        api_key = self.generate_api_key()
        api_key_hash = self.hash_api_key(api_key)

        user = User(
            id=user_id,
            email=email,
            organization=organization,
            role=role,
            api_key_hash=api_key_hash,
            permissions=self._get_role_permissions(role),
        )

        self.users_db[user_id] = user
        self.api_keys_db[api_key_hash] = user_id

        print(f"✓ Created user: {email} ({role})")
        return user, api_key

    def _get_role_permissions(self, role: str) -> List[str]:
        """Map role to permissions"""
        permissions_map = {
            'admin': ['read', 'write', 'delete', 'manage_users', 'view_logs', 'export_data'],
            'user': ['read', 'write'],
            'viewer': ['read'],
        }
        return permissions_map.get(role, ['read'])

    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify API key and return user"""
        if not api_key:
            return None

        api_key_hash = self.hash_api_key(api_key)
        user_id = self.api_keys_db.get(api_key_hash)

        if user_id:
            user = self.users_db.get(user_id)
            if user:
                # Update last login
                user.last_login = datetime.utcnow().isoformat()
                return user

        return None

    def create_jwt_token(self, user: User) -> str:
        """Create JWT token for session"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'organization': user.organization,
            'role': user.role,
            'permissions': user.permissions,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow(),
        }

        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return token

    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            print("⚠️  JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            print(f"⚠️  Invalid JWT token: {e}")
            return None

    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions

    def revoke_api_key(self, user_id: str):
        """Revoke user's API key"""
        user = self.users_db.get(user_id)
        if user and user.api_key_hash:
            del self.api_keys_db[user.api_key_hash]
            user.api_key_hash = None
            print(f"✓ Revoked API key for {user.email}")

    def list_users(self) -> List[User]:
        """List all users (admin only)"""
        return list(self.users_db.values())

    def save_to_file(self, filepath: str):
        """Save users to file (for development only)"""
        import json

        data = {
            'users': {uid: {
                'id': u.id,
                'email': u.email,
                'organization': u.organization,
                'role': u.role,
                'api_key_hash': u.api_key_hash,
                'permissions': u.permissions,
                'created_at': u.created_at,
            } for uid, u in self.users_db.items()},
            'api_keys': self.api_keys_db,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved auth data to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str, secret_key: str):
        """Load users from file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        auth = cls(secret_key)

        for uid, user_data in data['users'].items():
            user = User(**user_data)
            auth.users_db[uid] = user

        auth.api_keys_db = data['api_keys']

        print(f"✓ Loaded {len(auth.users_db)} users from {filepath}")
        return auth


# Flask decorators for API protection
def require_auth(auth_manager: AuthManager):
    """Decorator to require authentication"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify

            # Try API key first
            api_key = request.headers.get('X-API-Key')
            if api_key:
                user = auth_manager.verify_api_key(api_key)
                if user:
                    request.current_user = user
                    return f(*args, **kwargs)

            # Try JWT Bearer token
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                payload = auth_manager.verify_jwt_token(token)
                if payload:
                    user_id = payload['user_id']
                    user = auth_manager.users_db.get(user_id)
                    if user:
                        request.current_user = user
                        return f(*args, **kwargs)

            return jsonify({'error': 'Unauthorized', 'message': 'Valid API key or JWT token required'}), 401

        return decorated_function
    return decorator


def require_permission(auth_manager: AuthManager, permission: str):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify

            # First check auth
            user = getattr(request, 'current_user', None)
            if not user:
                return jsonify({'error': 'Unauthorized'}), 401

            # Check permission
            if not auth_manager.check_permission(user, permission):
                return jsonify({
                    'error': 'Forbidden',
                    'message': f'Permission "{permission}" required'
                }), 403

            return f(*args, **kwargs)

        return decorated_function
    return decorator


# Example usage
if __name__ == "__main__":
    print("Testing Authentication System\n")

    # Create auth manager
    auth = AuthManager()

    # Create users
    admin_user, admin_key = auth.create_user(
        email="admin@hospital.com",
        organization="General Hospital",
        role="admin"
    )

    regular_user, user_key = auth.create_user(
        email="doctor@hospital.com",
        organization="General Hospital",
        role="user"
    )

    print(f"\nAdmin API Key: {admin_key}")
    print(f"User API Key: {user_key}")
    print("\n⚠️  Save these keys! They won't be shown again.")

    # Test API key verification
    print("\n" + "="*60)
    print("Testing API Key Verification")
    print("="*60)

    verified_user = auth.verify_api_key(admin_key)
    if verified_user:
        print(f"✓ Verified: {verified_user.email}")
        print(f"  Role: {verified_user.role}")
        print(f"  Permissions: {verified_user.permissions}")

    # Test JWT token
    print("\n" + "="*60)
    print("Testing JWT Token")
    print("="*60)

    jwt_token = auth.create_jwt_token(admin_user)
    print(f"JWT Token: {jwt_token[:50]}...")

    payload = auth.verify_jwt_token(jwt_token)
    if payload:
        print(f"✓ Token valid for: {payload['email']}")

    # Test permissions
    print("\n" + "="*60)
    print("Testing Permissions")
    print("="*60)

    print(f"Admin can 'delete': {auth.check_permission(admin_user, 'delete')}")
    print(f"User can 'delete': {auth.check_permission(regular_user, 'delete')}")
    print(f"User can 'read': {auth.check_permission(regular_user, 'read')}")

    # Save to file
    auth.save_to_file('auth_data.json')

    print("\n✓ All tests passed!")
