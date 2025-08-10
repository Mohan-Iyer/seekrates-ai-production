# File: src/secrets/inject_llm_keys_from_aws.py
"""
Script DNA Header:
script_id: "fr_11_uc_091_ec_06_tc_250"
script_name: "src/secrets/inject_llm_keys_from_aws.py"
purpose: "Expose load_secrets() that fetches from AWS SM and exports envs (redacted logging)"
"""

# Auto-switches between real AWS injection and local test mode
# Requires: RUNTIME_ENV=[production|dev|test]

import os
import json
import sys
from typing import Optional, Tuple, Dict, List, Any

RUNTIME_ENV = os.getenv("RUNTIME_ENV", "dev").lower()

class SecretsInjector:
    def __init__(self, region_name: str = "ap-southeast-2"):
        self.region_name = region_name
        self.providers = ["OPENAI", "CLAUDE", "MISTRAL", "GEMINI", "COHERE", "AI21", "OLLAMA"]

        self.test_key = "sk-test-LOCAL-ONLY-1234567890"
        self.active_mode = "test" if RUNTIME_ENV == "test" else "aws"

        if self.active_mode == "aws":
            try:
                import boto3
                self.client = boto3.client("secretsmanager", region_name=self.region_name)
                print(f"✅ AWS Secrets Manager client initialized for region: {self.region_name}")
            except ImportError:
                raise RuntimeError("boto3 is required for production key injection")

    def inject_all_providers(self) -> Tuple[int, int]:
        injected = 0
        failed = 0

        if self.active_mode == "test":
            print("\n🚧 TEST MODE ACTIVE: Injecting fixed test key for all providers\n" + "=" * 70)
            for provider in self.providers:
                keyname = f"{provider}_API_KEY"
                os.environ[keyname] = self.test_key
                print(f"✅ {keyname} set to test value: {self.test_key}")
                injected += 1
            return (injected, failed)

        # AWS mode
        print(f"\n🚀 Starting LLM API key injection from AWS Secrets Manager\n📊 Total providers: {len(self.providers)}\n{'='*70}")
        for provider in self.providers:
            try:
                secret_id = f"seekrates_ai/{provider.lower()}_keys"
                response = self.client.get_secret_value(SecretId=secret_id)
                secret_string = response["SecretString"]

                os.environ[f"{provider}_API_KEY_ENC"] = secret_string
                print(f"✅ Injected {provider}_API_KEY_ENC from {secret_id}")
                injected += 1

            except Exception as e:
                print(f"❌ Failed to retrieve {provider}: {str(e)}")
                failed += 1

        return (injected, failed)

    def get_decrypted_key(self, provider: str) -> Optional[str]:
        if self.active_mode == "test":
            return self.test_key
        return os.getenv(f"{provider}_API_KEY_ENC")


def load_secrets() -> None:
    """
    Standardized AWS Secrets Manager bootstrap entrypoint
    
    Contract:
    - Reads AWS base vars + JSON config from environment
    - Fetches secrets from AWS SM, exports to env with redacted logging
    - Fail-fast if base AWS vars or REQUIRED_ENV_VARS missing post-bootstrap
    
    Environment inputs:
    - AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (required)
    - SECRET_PATHS (JSON list of secret names/ARNs, optional)
    - SECRET_KEY_MAP (JSON dict {secret_name: {source_key: ENV_VAR}}, optional)
    - REQUIRED_ENV_VARS (JSON list of env vars to verify post-bootstrap, optional)
    """
    
    print("🔐 AWS Secrets Bootstrap v1.0")
    
    # STEP 1: Validate AWS base environment variables
    aws_vars = ["AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing_aws = [var for var in aws_vars if not os.getenv(var)]
    
    if missing_aws:
        print(f"❌ ERR_AWS_ENV_MISSING: {missing_aws}")
        sys.exit(1)
    
    print(f"✅ AWS base vars present")
    
    # STEP 2: Parse JSON configuration inputs
    try:
        secret_paths_raw = os.getenv("SECRET_PATHS", "[]")
        secret_paths: List[str] = json.loads(secret_paths_raw)
        
        secret_key_map_raw = os.getenv("SECRET_KEY_MAP", "{}")
        secret_key_map: Dict[str, Dict[str, str]] = json.loads(secret_key_map_raw)
        
        required_env_vars_raw = os.getenv("REQUIRED_ENV_VARS", "[]")
        required_env_vars: List[str] = json.loads(required_env_vars_raw)
        
    except json.JSONDecodeError as e:
        print(f"❌ ERR_JSON_CONFIG_INVALID: {e}")
        sys.exit(2)
    
    print(f"📋 Config loaded: {len(secret_paths)} secrets, {len(required_env_vars)} required vars")
    
    # STEP 3: Choose strategy - specific secrets or provider injection
    if not secret_paths:
        # Use existing provider injection when no specific secrets specified
        print("🔑 Using provider injection strategy")
        try:
            injector = SecretsInjector(region_name=os.getenv("AWS_REGION"))
            injected, failed = injector.inject_all_providers()
            print(f"✅ Provider injection: {injected} success, {failed} failed")
        except Exception as e:
            print(f"❌ ERR_SECRET_FETCH_FAILED: provider injection - {e}")
            sys.exit(3)
    else:
        # Fetch specific secrets from SECRET_PATHS
        print("🎯 Using specific secrets strategy")
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION"))
            
            for secret_name in secret_paths:
                print(f"🔍 Fetching: {secret_name}")
                response = client.get_secret_value(SecretId=secret_name)
                secret_string = response["SecretString"]
                
                _process_mapped_secret(secret_name, secret_string, secret_key_map)
                print(f"✅ Fetched {secret_name} (redacted)")
                
        except Exception as e:
            print(f"❌ ERR_SECRET_FETCH_FAILED: {e}")
            sys.exit(3)
    
    # STEP 4: Verify required environment variables post-bootstrap
    if required_env_vars:
        missing_required = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_required:
            print(f"❌ ERR_REQUIRED_ENV_MISSING: {missing_required}")
            sys.exit(4)
        
        print(f"✅ Required vars verified: {len(required_env_vars)}")
    
    print("🎯 AWS secrets bootstrap completed")


def _process_mapped_secret(secret_name: str, secret_string: str, secret_key_map: Dict[str, Dict[str, str]]) -> None:
    """Process individual secret with optional key mapping"""
    
    if secret_name in secret_key_map:
        # Custom mapping specified
        mapping = secret_key_map[secret_name]
        
        try:
            secret_data = json.loads(secret_string)
            
            # Map JSON keys to environment variables per mapping
            for source_key, env_var in mapping.items():
                if source_key in secret_data:
                    value = str(secret_data[source_key])
                    os.environ[env_var] = value
                    print(f"  📌 Set {env_var}={_mask_value(value)}")
                else:
                    print(f"  ⚠️ Key '{source_key}' not found in secret")
                    
        except json.JSONDecodeError:
            # Secret is plain string, use first env var in mapping
            if mapping:
                env_var = list(mapping.values())[0]
                os.environ[env_var] = secret_string
                print(f"  📌 Set {env_var}={_mask_value(secret_string)}")
    else:
        # No mapping - auto-export simple JSON keys
        try:
            secret_data = json.loads(secret_string)
            
            for key, value in secret_data.items():
                if isinstance(value, (str, int, float, bool)):
                    env_var = key.upper()
                    os.environ[env_var] = str(value)
                    print(f"  📌 Set {env_var}={_mask_value(str(value))}")
                    
        except json.JSONDecodeError:
            print(f"  ⚠️ Plain string '{secret_name}' skipped (no mapping)")


def _mask_value(value: str) -> str:
    """Mask sensitive values for safe logging (xx****yy format)"""
    if not value or len(value) < 6:
        return "****"
    return f"{value[:2]}****{value[-2:]}"