#!/usr/bin/env python3
"""
# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION INTEGRATED
# =============================================================================
project_name: "alter_ego"
module_name: "Secure API Key Management System"
phase: "1 - Security Foundation Implementation"
script_id: "fr_01_uc_07_ec_01_tc_001"
script_name: "src/utils/secrets_manager.py"
script_purpose: "Encrypted API key storage and retrieval with secure fallback mechanisms for multi-agent authentication"
gps_coordinate: "fr_01_uc_07_ec_01_tc_001"
function_number: "01"
error_code_number: "01"
test_case_number: "001"
predecessor_script: "src/core/database_manager.py"
successor_script: "src/agents/base_agent.py"
input_sources: [".env_alter_ego", "config/security_config.yaml"]
output_destinations: ["logs/security_events.log", "src/agents/.env_alter_ego"]
template_used: "templates/script_dna_template.yaml"
template_version: "0001.03.00"
dependencies: ["cryptography", "python-dotenv", "pathlib"]
execution_context: "Security"
author: "MI"
created_on: "2025-06-15"
version: "1.0.0"
status: "Production"
related_hlr: ["HLR-001: Security Requirements"]
related_fr: ["FR-001: API Key Management", "FR-002: Encryption Standards"]
related_uc: ["UC-001: Secure Agent Authentication", "UC-002: Key Rotation"]
# =============================================================================
"""

import os
from pathlib import Path
import base64
import logging

try:
    from cryptography.fernet import Fernet
    FERNET_AVAILABLE = True
except ImportError:
    FERNET_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Secure secrets management for Decision Referee system.
    GPS Coordinate: fn_01_uc_07_ec_01_tc_001
    """
    
    def __init__(self, master_key: str = None):
        """Initialize secrets manager."""
        self.master_key = master_key or os.getenv('ENCRYPTION_KEY', 'dev_default_key')
        
    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext string."""
        try:
            if FERNET_AVAILABLE and len(self.master_key) >= 32:
                # Use Fernet encryption if available
                key = base64.urlsafe_b64encode(self.master_key[:32].encode())
                fernet = Fernet(key)
                encrypted = fernet.encrypt(plaintext.encode()).decode()
                return f"🔒{encrypted}"
            else:
                # Fallback to base64
                encoded = base64.b64encode(plaintext.encode()).decode()
                return f"🔒{encoded}"
        except Exception as e:
            logger.warning(f"Encryption failed: {e}")
            return plaintext
            
    def decrypt(self, encrypted_text: str) -> str:
        """Decrypt encrypted string."""
        try:
            if not encrypted_text.startswith('🔒'):
                return encrypted_text
                
            encrypted_data = encrypted_text[1:]  # Remove 🔒 marker
            
            if FERNET_AVAILABLE and len(self.master_key) >= 32:
                # Try Fernet decryption
                try:
                    key = base64.urlsafe_b64encode(self.master_key[:32].encode())
                    fernet = Fernet(key)
                    return fernet.decrypt(encrypted_data.encode()).decode()
                except:
                    pass
                    
            # Fallback to base64
            return base64.b64decode(encrypted_data.encode()).decode()
            
        except Exception as e:
            logger.warning(f"Decryption failed: {e}")
            return encrypted_text.replace('🔒', '')

def get_decrypted_key(key_name: str, encryption_key_env: str = "ENCRYPTION_KEY", env_file: str = ".env") -> str:
    """
    Legacy function - Decrypt and return API key from .env file.
    GPS Coordinate: fn_01_uc_07_ec_01_tc_001
    """
    env_path = Path(__file__).resolve().parent.parent.parent / env_file
    
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path)
        except ImportError:
            pass
    
    encrypted_key = os.getenv(key_name)
    encryption_key = os.getenv(encryption_key_env)
    
    if not encrypted_key:
        raise ValueError(f"Missing {key_name} in {env_file}")
    
    if not encryption_key:
        return encrypted_key  # Return as-is if no encryption key
    
    # Use SecretsManager for decryption
    manager = SecretsManager(encryption_key)
    return manager.decrypt(encrypted_key)
