# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "Utility Functions and Helpers"
# script_id: "fr_04_uc_406_ec_01_tc_406"
# gps_coordinate: "fr_04_uc_406_ec_01_tc_406"
# script_name: "llm_credentials_manager.py"
# template_version: "0002.00.00"
# status: "Production"
# =============================================================================

#!/usr/bin/env python3
# LLM LAW 3 COMPLIANCE: ENFORCED
# Verification: Yang - ChatGPT | Enforcement: 2025-06-29

from src.utils.path_resolver import get_path

#!/usr/bin/env python3
"""
# =============================================================================
# TEST SCRIPT HEADER - SIMPLIFIED METADATA
# =============================================================================
# NOTE: Test scripts do not require full Script DNA metadata per GPS Foundation
# guidelines. Tests are supporting infrastructure, not core business logic.
#
# Basic Test Identification:
# test_name: "test_llm_credentials_manager.py"
# target_script: "src/utils/llm_credentials_manager.py"
# target_gps_coordinate: "dp_01_uc_01_ec_01_tc_001"
# test_purpose: "Unit test coverage for LLM Credentials Manager"
# coding_engineer: "Claude"
# supervisor: "Yang - ChatGPT"
# test_framework: "pytest"
# =============================================================================
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import base64
from cryptography.fernet import Fernet

# Import the module under test
from src.utils.llm_credentials_manager import LLMCredentialsManager, load_llm_credentials


class TestLLMCredentialsManagerInit:
    """Test initialization of LLMCredentialsManager"""
    
    def test_init_default_env_path(self):
        """Test default .env path resolution"""
        with patch('src.utils.llm_credentials_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = Path("/project/root")
            
            manager = LLMCredentialsManager()
            
            assert manager.gps_coordinate == "dp_01_uc_01_ec_01_tc_001"
            assert manager.supervisor == "Yang - ChatGPT"
            assert manager.coding_engineer == "Claude"
            assert len(manager.supported_providers) == 7
            assert isinstance(manager.credentials, dict)
    
    def test_init_custom_env_path(self):
        """Test custom env file path"""
        custom_path = "/custom/path/.env"
        manager = LLMCredentialsManager(env_file_path=custom_path)
        
        assert str(manager.env_file_path) == custom_path
    
    @patch('src.utils.llm_credentials_manager.os.getenv')
    def test_init_encryption_key_from_env(self, mock_getenv):
        """Test encryption key loading from environment variable"""
        test_key = "test-encryption-key"
        mock_getenv.return_value = test_key
        
        manager = LLMCredentialsManager()
        
        assert manager.encryption_key == test_key.encode()
    
    @patch('src.utils.llm_credentials_manager.os.getenv')
    @patch('builtins.open', new_callable=mock_open, read_data=b'file-encryption-key')
    def test_init_encryption_key_from_file(self, mock_file, mock_getenv):
        """Test encryption key loading from file"""
        mock_getenv.return_value = None  # No env var
        
        with patch.object(Path, 'exists', return_value=True):
            manager = LLMCredentialsManager()
            assert manager.encryption_key == b'file-encryption-key'
    
    @patch('src.utils.llm_credentials_manager.os.getenv')
    @patch('src.utils.llm_credentials_manager.Fernet.generate_key')
    def test_init_encryption_key_generated(self, mock_generate, mock_getenv):
        """Test encryption key generation when none exists"""
        mock_getenv.return_value = None
        mock_generate.return_value = b'generated-key'
        
        with patch.object(Path, 'exists', return_value=False):
            manager = LLMCredentialsManager()
            assert manager.encryption_key == b'generated-key'
    
    def test_init_logging_setup(self):
        """Test GPS-coordinated logger creation"""
        manager = LLMCredentialsManager()
        
        assert manager.logger.name == "GPS:dp_01_uc_01_ec_01_tc_001"
    
    @patch.object(LLMCredentialsManager, '_emit_event')
    def test_init_event_emission(self, mock_emit):
        """Test initialization event tracking"""
        manager = LLMCredentialsManager()
        
        mock_emit.assert_called_once()
        args = mock_emit.call_args[0]
        assert args[0] == 'LLM_CREDENTIALS_MANAGER_INITIALIZED'
        assert 'gps_coordinate' in args[1]
        assert 'timestamp' in args[1]


class TestEnvironmentLoading:
    """Test environment file loading functionality"""
    
    def test_load_env_file_success(self):
        """Test valid .env file parsing"""
        env_content = """
# Comment line
OPENAI_API_KEY=sk-test123
CLAUDE_API_KEY="sk-ant-test456"
EMPTY_LINE=

# Another comment
COHERE_API_KEY='cohere-test789'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            
            manager = LLMCredentialsManager(env_file_path=f.name)
            manager._load_env_file()
            
            assert os.getenv('OPENAI_API_KEY') == 'sk-test123'
            assert os.getenv('CLAUDE_API_KEY') == 'sk-ant-test456'
            assert os.getenv('COHERE_API_KEY') == 'cohere-test789'
        
        os.unlink(f.name)
    
    def test_load_env_file_missing(self):
        """Test FileNotFoundError handling"""
        manager = LLMCredentialsManager(env_file_path="/nonexistent/.env")
        
        with pytest.raises(FileNotFoundError, match=".env file not found"):
            manager.load_all_credentials()
    
    def test_load_env_file_malformed(self):
        """Test handling of malformed .env content"""
        env_content = """
VALID_KEY=valid_value
MALFORMED_LINE_NO_EQUALS
=MISSING_KEY
KEY_WITH_MULTIPLE=EQUALS=SIGNS
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            
            manager = LLMCredentialsManager(env_file_path=f.name)
            manager._load_env_file()
            
            # Should still load valid entries
            assert os.getenv('VALID_KEY') == 'valid_value'
            assert os.getenv('KEY_WITH_MULTIPLE') == 'EQUALS=SIGNS'
        
        os.unlink(f.name)
    
    def test_load_env_file_comments_and_empty_lines(self):
        """Test comment and empty line handling"""
        env_content = """
# This is a comment
TEST_KEY=test_value

# Another comment with spaces
   # Indented comment
TEST_KEY2=test_value2

"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            
            manager = LLMCredentialsManager(env_file_path=f.name)
            manager._load_env_file()
            
            assert os.getenv('TEST_KEY') == 'test_value'
            assert os.getenv('TEST_KEY2') == 'test_value2'
        
        os.unlink(f.name)
    
    def test_load_env_file_quote_stripping(self):
        """Test quote stripping validation"""
        env_content = '''
SINGLE_QUOTES='single_value'
DOUBLE_QUOTES="double_value"
NO_QUOTES=no_quotes_value
MIXED_QUOTES="mixed'quotes"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            
            manager = LLMCredentialsManager(env_file_path=f.name)
            manager._load_env_file()
            
            assert os.getenv('SINGLE_QUOTES') == 'single_value'
            assert os.getenv('DOUBLE_QUOTES') == 'double_value'
            assert os.getenv('NO_QUOTES') == 'no_quotes_value'
            assert os.getenv('MIXED_QUOTES') == "mixed'quotes"
        
        os.unlink(f.name)


class TestCredentialDecryption:
    """Test credential decryption functionality"""
    
    def test_decrypt_fernet_format(self):
        """Test gAAAAAB prefix decryption"""
        manager = LLMCredentialsManager()
        test_key = Fernet.generate_key()
        manager.encryption_key = test_key
        
        fernet = Fernet(test_key)
        original_value = "sk-test123456789"
        encrypted_value = fernet.encrypt(original_value.encode()).decode()
        
        decrypted = manager.decrypt_env_var(encrypted_value)
        assert decrypted == original_value
    
    def test_decrypt_custom_base64(self):
        """Test ENC_ prefix decryption"""
        manager = LLMCredentialsManager()
        manager.encryption_key = b'test-key'
        
        original_value = "sk-test123456789"
        encoded_value = base64.b64encode(original_value.encode()).decode()
        encrypted_value = f"ENC_{encoded_value}"
        
        decrypted = manager.decrypt_env_var(encrypted_value)
        assert decrypted == original_value
    
    def test_decrypt_plaintext_passthrough(self):
        """Test plaintext handling"""
        manager = LLMCredentialsManager()
        plaintext_value = "sk-plaintext123"
        
        decrypted = manager.decrypt_env_var(plaintext_value)
        assert decrypted == plaintext_value
    
    def test_decrypt_invalid_encryption(self):
        """Test malformed encrypted data handling"""
        manager = LLMCredentialsManager()
        manager.encryption_key = Fernet.generate_key()
        
        invalid_encrypted = "gAAAAABinvalid_data"
        
        # Should return original value if decryption fails
        decrypted = manager.decrypt_env_var(invalid_encrypted)
        assert decrypted == invalid_encrypted
    
    def test_decrypt_no_encryption_key(self):
        """Test handling when no encryption key is available"""
        manager = LLMCredentialsManager()
        manager.encryption_key = None
        
        encrypted_value = "gAAAAABsomeencrypteddata"
        decrypted = manager.decrypt_env_var(encrypted_value)
        
        # Should return original value
        assert decrypted == encrypted_value
    
    def test_decrypt_key_mismatch(self):
        """Test handling when wrong key is used"""
        manager = LLMCredentialsManager()
        
        # Encrypt with one key
        key1 = Fernet.generate_key()
        fernet1 = Fernet(key1)
        encrypted_value = fernet1.encrypt(b"test_value").decode()
        
        # Try to decrypt with different key
        key2 = Fernet.generate_key()
        manager.encryption_key = key2
        
        decrypted = manager.decrypt_env_var(encrypted_value)
        # Should return original encrypted value on failure
        assert decrypted == encrypted_value


class TestProviderAvailability:
    """Test provider availability functionality"""
    
    def test_all_providers_available(self):
        """Test when all 7 providers are loaded"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-openai123',
            'CLAUDE_API_KEY': 'sk-ant-claude123',
            'COHERE_API_KEY': 'cohere123',
            'MISTRAL_API_KEY': 'mistral123',
            'AI21_API_KEY': 'ai21123',
            'TOGETHER_API_KEY': 'together123',
            'FIREWORKS_API_KEY': 'fireworks123'
        }):
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(LLMCredentialsManager, '_load_env_file'):
                    manager = LLMCredentialsManager()
                    credentials = manager.load_all_credentials()
                    
                    assert len(credentials) == 7
                    assert len(manager.get_available_providers()) == 7
    
    def test_partial_providers_available(self):
        """Test when some providers are missing"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-openai123',
            'CLAUDE_API_KEY': 'sk-ant-claude123',
            # Missing other providers
        }, clear=True):
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(LLMCredentialsManager, '_load_env_file'):
                    manager = LLMCredentialsManager()
                    credentials = manager.load_all_credentials()
                    
                    assert len(credentials) == 2
                    assert 'OPENAI_API_KEY' in credentials
                    assert 'CLAUDE_API_KEY' in credentials
    
    def test_no_providers_available(self):
        """Test when no providers are available"""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(LLMCredentialsManager, '_load_env_file'):
                    manager = LLMCredentialsManager()
                    credentials = manager.load_all_credentials()
                    
                    assert len(credentials) == 0
                    assert len(manager.get_available_providers()) == 0
    
    def test_get_credential_existing(self):
        """Test valid provider key lookup"""
        manager = LLMCredentialsManager()
        manager.credentials = {'OPENAI_API_KEY': 'sk-test123'}
        
        credential = manager.get_credential('OPENAI_API_KEY')
        assert credential == 'sk-test123'
    
    def test_get_credential_missing(self):
        """Test non-existent provider key"""
        manager = LLMCredentialsManager()
        manager.credentials = {}
        
        credential = manager.get_credential('NONEXISTENT_KEY')
        assert credential is None
    
    def test_is_provider_available_true(self):
        """Test available provider check"""
        manager = LLMCredentialsManager()
        manager.credentials = {'OPENAI_API_KEY': 'sk-test123'}
        
        assert manager.is_provider_available('OPENAI_API_KEY') is True
    
    def test_is_provider_available_false(self):
        """Test unavailable provider check"""
        manager = LLMCredentialsManager()
        manager.credentials = {}
        
        assert manager.is_provider_available('OPENAI_API_KEY') is False
    
    def test_get_available_providers_mapping(self):
        """Test available providers mapping"""
        manager = LLMCredentialsManager()
        manager.credentials = {
            'OPENAI_API_KEY': 'sk-test123',
            'CLAUDE_API_KEY': 'sk-ant-test456'
        }
        
        available = manager.get_available_providers()
        expected = {
            'OPENAI_API_KEY': 'OpenAI GPT Models',
            'CLAUDE_API_KEY': 'Anthropic Claude Models'
        }
        assert available == expected


class TestCredentialValidation:
    """Test credential format validation"""
    
    def test_validate_openai_format(self):
        """Test OpenAI sk- prefix validation"""
        manager = LLMCredentialsManager()
        manager.credentials = {
            'OPENAI_API_KEY': 'sk-valid123456789',
        }
        
        validation = manager.validate_credentials()
        assert validation['OPENAI_API_KEY'] is True
    
    def test_validate_claude_format(self):
        """Test Claude sk-ant- prefix validation"""
        manager = LLMCredentialsManager()
        manager.credentials = {
            'CLAUDE_API_KEY': 'sk-ant-valid123456789',
        }
        
        validation = manager.validate_credentials()
        assert validation['CLAUDE_API_KEY'] is True
    
    def test_validate_generic_format(self):
        """Test length-based validation for other providers"""
        manager = LLMCredentialsManager()
        manager.credentials = {
            'COHERE_API_KEY': 'cohere-valid-key-12345',
        }
        
        validation = manager.validate_credentials()
        assert validation['COHERE_API_KEY'] is True
    
    def test_validate_invalid_formats(self):
        """Test invalid credential formats"""
        manager = LLMCredentialsManager()
        manager.credentials = {
            'OPENAI_API_KEY': 'invalid-key',  # Should start with sk-
            'CLAUDE_API_KEY': 'sk-wrong',     # Should start with sk-ant-
            'COHERE_API_KEY': 'short',        # Too short
        }
        
        validation = manager.validate_credentials()
        assert validation['OPENAI_API_KEY'] is False
        assert validation['CLAUDE_API_KEY'] is False
        assert validation['COHERE_API_KEY'] is False
    
    def test_validate_empty_credentials(self):
        """Test empty credential handling"""
        manager = LLMCredentialsManager()
        manager.credentials = {}
        
        validation = manager.validate_credentials()
        assert validation == {}
    
    def test_validate_mixed_validity(self):
        """Test mix of valid and invalid credentials"""
        manager = LLMCredentialsManager()
        manager.credentials = {
            'OPENAI_API_KEY': 'sk-validkey123',
            'CLAUDE_API_KEY': 'invalid-claude',
            'COHERE_API_KEY': 'cohere-long-valid-key',
        }
        
        validation = manager.validate_credentials()
        assert validation['OPENAI_API_KEY'] is True
        assert validation['CLAUDE_API_KEY'] is False
        assert validation['COHERE_API_KEY'] is True


class TestErrorSimulation:
    """Test error handling and GPS error capture"""
    
    @patch.object(LLMCredentialsManager, '_capture_gps_error')
    def test_gps_error_capture(self, mock_capture):
        """Test GPS error recording"""
        manager = LLMCredentialsManager(env_file_path="/nonexistent/.env")
        
        with pytest.raises(FileNotFoundError):
            manager.load_all_credentials()
        
        mock_capture.assert_called_once()
        args = mock_capture.call_args[1]
        assert args['gps_coordinate'] == 'dp_01_uc_01_ec_01_tc_001_load'
        assert 'Failed to load LLM credentials' in args['error_message']
        assert args['error_type'] == 'credential_loading_error'
    
    def test_file_permission_error(self):
        """Test handling of file permission errors"""
        # Create a file with restricted permissions
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"OPENAI_API_KEY=sk-test123")
            f.flush()
            os.chmod(f.name, 0o000)  # No permissions
            
            manager = LLMCredentialsManager(env_file_path=f.name)
            
            with pytest.raises(Exception):
                manager.load_all_credentials()
        
        # Cleanup
        os.chmod(f.name, 0o666)
        os.unlink(f.name)
    
    @patch('src.utils.llm_credentials_manager.Fernet')
    def test_encryption_library_error(self, mock_fernet):
        """Test handling of cryptography library errors"""
        mock_fernet.side_effect = ImportError("Cryptography not available")
        
        with pytest.raises(ImportError):
            manager = LLMCredentialsManager()
    
    def test_corrupted_env_file(self):
        """Test handling of corrupted/binary env file"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'\x00\x01\x02\x03')  # Binary data
            f.flush()
            
            manager = LLMCredentialsManager(env_file_path=f.name)
            
            # Should handle binary data gracefully
            try:
                manager._load_env_file()
            except Exception as e:
                assert "Failed to load .env file" in str(e)
        
        os.unlink(f.name)


class TestEventEmission:
    """Test event emission functionality"""
    
    @patch.object(LLMCredentialsManager, '_emit_event')
    def test_initialization_event(self, mock_emit):
        """Test manager initialization event"""
        manager = LLMCredentialsManager()
        
        mock_emit.assert_called_once_with(
            'LLM_CREDENTIALS_MANAGER_INITIALIZED',
            {
                'gps_coordinate': 'dp_01_uc_01_ec_01_tc_001',
                'env_file_path': str(manager.env_file_path),
                'supported_providers': 7,
                'encryption_enabled': manager.encryption_key is not None,
                'timestamp': mock_emit.call_args[0][1]['timestamp']
            }
        )
    
    @patch.object(LLMCredentialsManager, '_emit_event')
    def test_credentials_loaded_event(self, mock_emit):
        """Test successful credential load event"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'}):
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(LLMCredentialsManager, '_load_env_file'):
                    manager = LLMCredentialsManager()
                    mock_emit.reset_mock()  # Reset to ignore init event
                    
                    manager.load_all_credentials()
                    
                    # Should have called emit for credentials loaded
                    assert mock_emit.call_count >= 1
                    last_call = mock_emit.call_args_list[-1]
                    assert last_call[0][0] == 'LLM_CREDENTIALS_LOADED'
    
    def test_event_data_structure(self):
        """Test event payload validation"""
        manager = LLMCredentialsManager()
        
        with patch.object(manager.logger, 'info') as mock_log:
            manager._emit_event('TEST_EVENT', {'key': 'value'})
            
            mock_log.assert_called_once()
            log_message = mock_log.call_args[0][0]
            assert 'EVENT:TEST_EVENT' in log_message
            assert 'key' in log_message
    
    def test_event_timestamp_format(self):
        """Test ISO timestamp format in events"""
        manager = LLMCredentialsManager()
        
        with patch.object(manager.logger, 'info') as mock_log:
            test_data = {'timestamp': '2025-06-22T09:00:00.000000'}
            manager._emit_event('TEST_EVENT', test_data)
            
            mock_log.assert_called_once()


class TestUtilityFunction:
    """Test standalone utility function"""
    
    @patch('src.utils.llm_credentials_manager.LLMCredentialsManager')
    def test_load_llm_credentials_success(self, mock_manager_class):
        """Test utility function success path"""
        mock_manager = Mock()
        mock_manager.load_all_credentials.return_value = {'OPENAI_API_KEY': 'sk-test123'}
        mock_manager_class.return_value = mock_manager
        
        result = load_llm_credentials()
        
        mock_manager_class.assert_called_once_with(None)
        mock_manager.load_all_credentials.assert_called_once()
        assert result == {'OPENAI_API_KEY': 'sk-test123'}
    
    @patch('src.utils.llm_credentials_manager.LLMCredentialsManager')
    def test_load_llm_credentials_custom_path(self, mock_manager_class):
        """Test utility function with custom env path"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        custom_path = get_path("/custom/.env")
        
        load_llm_credentials(env_file_path=custom_path)
        
        mock_manager_class.assert_called_once_with(custom_path)
    
    @patch('src.utils.llm_credentials_manager.LLMCredentialsManager')
    def test_load_llm_credentials_failure(self, mock_manager_class):
        """Test utility function error handling"""
        mock_manager = Mock()
        mock_manager.load_all_credentials.side_effect = Exception("Test error")
        mock_manager_class.return_value = mock_manager
        
        with pytest.raises(Exception, match="Test error"):
            load_llm_credentials()


# Test fixtures for reuse
@pytest.fixture
def mock_env_file():
    """Create temporary .env file for testing"""
    content = """
OPENAI_API_KEY=sk-test123456789
CLAUDE_API_KEY=sk-ant-test456789
COHERE_API_KEY=cohere-test-key-123
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_encrypted_credentials():
    """Sample encrypted credential data"""
    key = Fernet.generate_key()
    fernet = Fernet(key)
    return {
        'key': key,
        'encrypted_openai': fernet.encrypt(b'sk-real-openai-key').decode(),
        'encrypted_claude': fernet.encrypt(b'sk-ant-real-claude-key').decode(),
    }


@pytest.fixture
def mock_encryption_key():
    """Test encryption key"""
    return Fernet.generate_key()


@pytest.fixture
def credentials_manager():
    """Pre-configured manager instance for testing"""
    with patch.object(LLMCredentialsManager, '_load_encryption_key'):
        manager = LLMCredentialsManager()
        manager.encryption_key = Fernet.generate_key()
        return manager


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])