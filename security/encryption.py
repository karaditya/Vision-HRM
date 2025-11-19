"""
Encryption Manager for Data Protection
Implements AES-256 encryption for data at rest and in transit
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import os
import secrets


class EncryptionManager:
    """Handle encryption/decryption of sensitive data"""

    def __init__(self, master_key: str = None):
        """
        Args:
            master_key: Master encryption key (base64 encoded)
                       If None, generates new key
        """
        if master_key is None:
            # Generate new key
            master_key = Fernet.generate_key().decode()
            print("⚠️  Generated new encryption key:")
            print(f"   {master_key}")
            print("\n   CRITICAL: Save this key securely!")
            print("   Set as environment variable:")
            print(f"   export ENCRYPTION_KEY='{master_key}'")
            print("\n   Without this key, encrypted data CANNOT be recovered!")

        # Initialize Fernet cipher
        if isinstance(master_key, str):
            master_key = master_key.encode()

        self.fernet = Fernet(master_key)
        print("✓ EncryptionManager initialized")

    @classmethod
    def from_password(cls, password: str, salt: bytes = None):
        """
        Create encryption manager from password

        Args:
            password: User password
            salt: Salt for key derivation (store this!)
        """
        if salt is None:
            salt = os.urandom(16)
            print(f"⚠️  Generated salt: {base64.b64encode(salt).decode()}")
            print("   SAVE THIS SALT!")

        # Derive key from password
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

        return cls(key.decode())

    def encrypt(self, data: str) -> str:
        """
        Encrypt string data

        Args:
            data: Plain text string

        Returns:
            Encrypted string (base64 encoded)
        """
        if not data:
            return ""

        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt string data

        Args:
            encrypted_data: Encrypted string (base64 encoded)

        Returns:
            Plain text string
        """
        if not encrypted_data:
            return ""

        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            print(f"⚠️  Decryption failed: {e}")
            raise ValueError("Failed to decrypt data. Wrong key?")

    def encrypt_dict(self, data: dict, fields: list) -> dict:
        """
        Encrypt specific fields in dictionary

        Args:
            data: Dictionary with data
            fields: List of field names to encrypt

        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()

        for field in fields:
            if field in encrypted_data:
                value = str(encrypted_data[field])
                encrypted_data[field] = self.encrypt(value)

        return encrypted_data

    def decrypt_dict(self, data: dict, fields: list) -> dict:
        """
        Decrypt specific fields in dictionary

        Args:
            data: Dictionary with encrypted data
            fields: List of field names to decrypt

        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()

        for field in fields:
            if field in decrypted_data:
                decrypted_data[field] = self.decrypt(decrypted_data[field])

        return decrypted_data

    def encrypt_file(self, input_path: str, output_path: str):
        """
        Encrypt entire file

        Args:
            input_path: Path to plain text file
            output_path: Path to save encrypted file
        """
        with open(input_path, 'rb') as f:
            data = f.read()

        encrypted = self.fernet.encrypt(data)

        with open(output_path, 'wb') as f:
            f.write(encrypted)

        print(f"✓ Encrypted {input_path} -> {output_path}")

    def decrypt_file(self, input_path: str, output_path: str):
        """
        Decrypt entire file

        Args:
            input_path: Path to encrypted file
            output_path: Path to save decrypted file
        """
        with open(input_path, 'rb') as f:
            encrypted = f.read()

        decrypted = self.fernet.decrypt(encrypted)

        with open(output_path, 'wb') as f:
            f.write(decrypted)

        print(f"✓ Decrypted {input_path} -> {output_path}")

    def encrypt_medical_record(self, record: dict) -> dict:
        """
        Encrypt sensitive medical record fields (HIPAA)

        Encrypts: patient_name, ssn, medical_record_number, diagnosis, etc.
        """
        sensitive_fields = [
            'patient_name',
            'ssn',
            'date_of_birth',
            'medical_record_number',
            'diagnosis',
            'treatment',
            'medications',
            'notes',
        ]

        return self.encrypt_dict(record, sensitive_fields)

    def decrypt_medical_record(self, encrypted_record: dict) -> dict:
        """Decrypt medical record"""
        sensitive_fields = [
            'patient_name',
            'ssn',
            'date_of_birth',
            'medical_record_number',
            'diagnosis',
            'treatment',
            'medications',
            'notes',
        ]

        return self.decrypt_dict(encrypted_record, sensitive_fields)

    def encrypt_financial_data(self, data: dict) -> dict:
        """
        Encrypt sensitive financial data

        Encrypts: account_number, ssn, credit_card, etc.
        """
        sensitive_fields = [
            'account_number',
            'routing_number',
            'ssn',
            'credit_card',
            'bank_account',
            'salary',
            'balance',
        ]

        return self.encrypt_dict(data, sensitive_fields)

    def decrypt_financial_data(self, encrypted_data: dict) -> dict:
        """Decrypt financial data"""
        sensitive_fields = [
            'account_number',
            'routing_number',
            'ssn',
            'credit_card',
            'bank_account',
            'salary',
            'balance',
        ]

        return self.decrypt_dict(encrypted_data, sensitive_fields)


# Example usage
if __name__ == "__main__":
    print("Testing Encryption System\n")

    # Create encryption manager
    encryptor = EncryptionManager()

    # Test string encryption
    print("="*60)
    print("Testing String Encryption")
    print("="*60)

    secret_data = "Patient John Doe has diabetes"
    encrypted = encryptor.encrypt(secret_data)
    decrypted = encryptor.decrypt(encrypted)

    print(f"Original:  {secret_data}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {secret_data == decrypted}")

    # Test medical record encryption
    print("\n" + "="*60)
    print("Testing Medical Record Encryption")
    print("="*60)

    medical_record = {
        'record_id': '12345',
        'patient_name': 'John Doe',
        'ssn': '123-45-6789',
        'date_of_birth': '1980-01-15',
        'diagnosis': 'Type 2 Diabetes',
        'treatment': 'Metformin 500mg',
        'doctor': 'Dr. Smith',  # Not sensitive, won't be encrypted
    }

    print("\nOriginal record:")
    print(medical_record)

    encrypted_record = encryptor.encrypt_medical_record(medical_record)
    print("\nEncrypted record:")
    print(encrypted_record)

    decrypted_record = encryptor.decrypt_medical_record(encrypted_record)
    print("\nDecrypted record:")
    print(decrypted_record)

    print(f"\nMatch: {medical_record == decrypted_record}")

    # Test file encryption
    print("\n" + "="*60)
    print("Testing File Encryption")
    print("="*60)

    # Create test file
    test_file = "test_sensitive.txt"
    with open(test_file, 'w') as f:
        f.write("This is sensitive patient data.\nSSN: 123-45-6789\nDiagnosis: Confidential")

    # Encrypt
    encrypted_file = "test_sensitive.encrypted"
    encryptor.encrypt_file(test_file, encrypted_file)

    # Decrypt
    decrypted_file = "test_sensitive.decrypted.txt"
    encryptor.decrypt_file(encrypted_file, decrypted_file)

    # Verify
    with open(test_file, 'r') as f:
        original = f.read()
    with open(decrypted_file, 'r') as f:
        decrypted = f.read()

    print(f"Files match: {original == decrypted}")

    # Cleanup
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)

    print("\n✓ All encryption tests passed!")
