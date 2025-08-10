# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "seekrates_ai"
# module_name: "Agent Security Guard"
# script_id: "fr_06_uc_01_ec_01_tc_301"
# gps_coordinate: "fr_06_uc_01_ec_01_tc_301"
# script_name: "src/secrets/secure_runtime_guard.py"
# template_version: "0003.00.00"
# status: "Production"
# =============================================================================

"""
Secure Runtime Guard - Multi-Layer Agent Protection
Prevents unauthorized deployment of seekrates_ai agent system through:
1. AWS IAM Identity Verification
2. Fernet Key Fingerprint Binding  
3. Host/Deployment Fingerprint Validation
4. LLM API Key Environment Validation
"""

import os
import sys
import uuid
import platform
import socket
import hashlib
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# =============================================================================
# SCRIPT DNA STRUCTURE
# =============================================================================

# CORE IDENTIFICATION
project_name: str = "seekrates_ai"
module_name: str = "Agent Security Guard"
phase: str = "6 - Production Security Validation"
script_id: str = "fr_06_uc_01_ec_01_tc_301"
script_name: str = "src/secrets/secure_runtime_guard.py"
script_purpose: str = "Multi-layer security validation preventing unauthorized deployment of monetizable AI agent system"

# GPS FOUNDATION INTEGRATION
gps_coordinate: str = "fr_06_uc_01_ec_01_tc_301"
function_number: str = "fn_06"
error_code_number: str = "ec_01"
test_case_number: str = "tc_301"

# =============================================================================
# OBJECT MODEL METADATA (MANDATORY)
# =============================================================================
object_model = {
    "defines_classes": [
        {
            "name": "SecureRuntimeGuard",
            "purpose": "Multi-layer security validation for agent system protection",
            "interfaces_implemented": [],
            "inheritance": []
        }
    ],
    "uses_classes": [
        {
            "name": "boto3.client",
            "source_module": "boto3",
            "relationship": "composition",
            "purpose": "AWS STS client for identity verification"
        }
    ],
    "instantiates": [
        {
            "class_name": "SecureRuntimeGuard",
            "instance_name": "guard",
            "scope": "module",
            "lifecycle": "created_at_runtime|destroyed_at_validation_complete"
        }
    ],
    "depends_on_interfaces": [
        {
            "interface_name": "AWS_STS_Service",
            "provided_by": "AWS_IAM",
            "required_methods": ["get_caller_identity"],
            "dependency_type": "service_injection"
        }
    ],
    "collaborations": [
        {
            "collaborator": "sys.exit",
            "relationship": "calls",
            "interaction_pattern": "command",
            "data_flow_direction": "to"
        }
    ]
}

# TRACEABILITY CHAIN
predecessor_script: str = "src/secrets/inject_llm_keys_from_aws.py"
successor_script: str = "src/agents/multi_llm_agent_launcher.py"
predecessor_template: str = "templates/script_dna_template.yaml"
successor_template: str = "templates/script_dna_template.yaml"

# INPUT/OUTPUT SPECIFICATIONS
input_sources = [
    {
        "path": ".env",
        "type": "Environment_File",
        "description": "Fernet key and deployment configuration",
        "providing_object": "FileSystem",
        "verification_status": "verified_exists"
    },
    {
        "path": "AWS_STS_Service",
        "type": "AWS_Service",
        "description": "AWS identity verification service",
        "providing_object": "AWS_IAM",
        "verification_status": "verified_exists"
    }
]

output_destinations = [
    {
        "path": "sys.stdout",
        "type": "Console_Output",
        "description": "Security validation results and authorization status",
        "consuming_object": "Runtime_Console",
        "verification_status": "verified_exists"
    }
]

# TEMPLATE LINEAGE
template_used: str = "templates/script_dna_template.yaml"
template_version: str = "0003.00.00"
generation_timestamp: str = "2025-07-22T16:00:00Z"

# DEPENDENCY MAPPING
dependencies = {
    "internal_modules": [],
    "external_libraries": ["boto3", "botocore", "python-dotenv", "hashlib", "uuid", "platform", "socket"],
    "database_connections": [],
    "redis_connections": [],
    "interface_dependencies": [
        {
            "interface": "AWS_STS_Service",
            "implementation": "AWS_IAM",
            "required_by": "SecureRuntimeGuard.verify_aws_identity",
            "verification_status": "verified"
        }
    ]
}

# EXECUTION CONTEXT
execution_context: str = "CLI"
runtime_environment: str = "Production"
session_mode: str = "memoryless"
memory_source: str = "None"

# VERSIONING & AUTHORSHIP
author: str = "MI"
created_on: str = "2025-07-22T16:00:00Z"
last_updated: str = "2025-07-22T16:00:00Z"
version: str = "1.0.0"
status: str = "Production"
coding_engineer: str = "Claude"
supervisor: str = "Yang - ChatGPT"
business_owner: str = "Mohan Iyer mohan@pixels.net.nz"

# MIGRATION SUPPORT
migration = {
    "suggested_category": "core",
    "migratable": False,
    "reason_if_not_migratable": "Contains security fingerprints specific to authorized deployment environment",
    "migration_strategy": "exclude",
    "destination_mapping": "N/A"
}

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# AWS Account Security - Replace with your actual trusted account ID
TRUSTED_AWS_ACCOUNT_ID = "382578233801"  # Your actual account ID from earlier

# Required encrypted LLM API key environment variables
REQUIRED_API_KEYS: List[str] = [
    "OPENAI_API_KEY_ENC",
    "CLAUDE_API_KEY_ENC", 
    "MISTRAL_API_KEY_ENC",
    "GEMINI_API_KEY_ENC",
    "COHERE_API_KEY_ENC",
    "AI21_API_KEY_ENC",
    "OLLAMA_API_KEY_ENC"
]

# Authorized deployment fingerprints - GENERATED FOR THIS ENVIRONMENT
AUTHORIZED_FERNET_FINGERPRINTS = [
    "d1db162debfa9fb4f7eaf03ec538e9f63906494838117bb92395bfa706b7e2f8",
]

AUTHORIZED_DEPLOYMENT_FINGERPRINTS = [
    "e8897ebeb000c86b1a0f2b6f71867cc629d7afac5e804556853bf0ed8aa54012",
]

# Security validation settings
MIN_KEY_LENGTH = 10
EXPECTED_KEY_PREFIX = "🔒"

# =============================================================================
# MAIN IMPLEMENTATION
# =============================================================================

class SecureRuntimeGuard:
    """
    Multi-layer security validation for agent system protection
    """
    
    def __init__(self):
        """
        Initialize the security guard with multi-layer protection
        """
        self.required_keys = REQUIRED_API_KEYS
        self.validation_results: Dict[str, bool] = {}
        self.security_failures: List[str] = []
        
        # Load environment variables
        load_dotenv()
        
    def verify_aws_identity(self) -> Tuple[bool, str]:
        """
        Verify AWS account identity matches trusted account
        
        Returns:
            Tuple of (is_authorized, status_message)
        """
        try:
            sts_client = boto3.client('sts')
            identity = sts_client.get_caller_identity()
            
            account_id = identity.get('Account')
            user_arn = identity.get('Arn', 'Unknown')
            
            if account_id == TRUSTED_AWS_ACCOUNT_ID:
                return True, f"✅ AWS_IDENTITY: Authorized account {account_id} | {user_arn}"
            else:
                return False, f"❌ AWS_IDENTITY: Unauthorized account {account_id} (expected {TRUSTED_AWS_ACCOUNT_ID})"
                
        except NoCredentialsError:
            return False, "❌ AWS_IDENTITY: No AWS credentials found"
        except ClientError as e:
            return False, f"❌ AWS_IDENTITY: AWS error - {str(e)}"
        except Exception as e:
            return False, f"❌ AWS_IDENTITY: Validation error - {str(e)}"
    
    def generate_fernet_fingerprint(self) -> Optional[str]:
        """
        Generate fingerprint from Fernet key and machine identity
        
        Returns:
            SHA256 fingerprint or None if error
        """
        try:
            fernet_key = os.getenv('FERNET_KEY')
            if not fernet_key:
                return None
            
            # Machine identity components
            node_id = str(uuid.getnode())
            hostname = platform.node()
            
            # Create composite fingerprint
            composite = f"{fernet_key}|{node_id}|{hostname}"
            fingerprint = hashlib.sha256(composite.encode()).hexdigest()
            
            return fingerprint
            
        except Exception:
            return None
    
    def verify_fernet_fingerprint(self) -> Tuple[bool, str]:
        """
        Verify Fernet key fingerprint matches authorized deployment
        
        Returns:
            Tuple of (is_authorized, status_message)
        """
        fernet_key = os.getenv('FERNET_KEY')
        if not fernet_key:
            return False, "❌ FERNET_CHECK: FERNET_KEY not found in environment"
        
        current_fingerprint = self.generate_fernet_fingerprint()
        if not current_fingerprint:
            return False, "❌ FERNET_CHECK: Failed to generate fingerprint"
        
        # Check against authorized fingerprints
        if current_fingerprint in AUTHORIZED_FERNET_FINGERPRINTS:
            masked_fingerprint = current_fingerprint[:16] + "..."
            return True, f"✅ FERNET_CHECK: Authorized fingerprint {masked_fingerprint}"
        else:
            masked_fingerprint = current_fingerprint[:16] + "..."
            return False, f"❌ FERNET_CHECK: Unauthorized fingerprint {masked_fingerprint}"
    
    def generate_deployment_fingerprint(self) -> str:
        """
        Generate deployment environment fingerprint
        
        Returns:
            SHA256 fingerprint of deployment environment
        """
        try:
            # Multiple environment identifiers
            node_id = str(uuid.getnode())
            hostname = socket.gethostname()
            platform_info = f"{platform.system()}-{platform.machine()}"
            
            # Include deployment ID if provided
            deployment_id = os.getenv('DEPLOYMENT_ID', 'default')
            device_id = os.getenv('DEVICE_ID', 'default')
            
            # Create composite identifier
            composite = f"{node_id}|{hostname}|{platform_info}|{deployment_id}|{device_id}"
            fingerprint = hashlib.sha256(composite.encode()).hexdigest()
            
            return fingerprint
            
        except Exception as e:
            # Fallback fingerprint
            return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()
    
    def verify_deployment_fingerprint(self) -> Tuple[bool, str]:
        """
        Verify deployment environment fingerprint
        
        Returns:
            Tuple of (is_authorized, status_message)
        """
        current_fingerprint = self.generate_deployment_fingerprint()
        
        # Check against authorized deployment fingerprints
        if current_fingerprint in AUTHORIZED_DEPLOYMENT_FINGERPRINTS:
            masked_fingerprint = current_fingerprint[:16] + "..."
            return True, f"✅ DEPLOYMENT_CHECK: Authorized environment {masked_fingerprint}"
        else:
            masked_fingerprint = current_fingerprint[:16] + "..."
            return False, f"❌ DEPLOYMENT_CHECK: Unauthorized environment {masked_fingerprint}"
    
    def validate_api_keys(self) -> Tuple[bool, List[str]]:
        """
        Validate all required LLM API keys are present
        
        Returns:
            Tuple of (all_valid, validation_messages)
        """
        messages = []
        all_valid = True
        
        for key_name in self.required_keys:
            if key_name not in os.environ:
                messages.append(f"❌ API_KEY: {key_name} missing")
                all_valid = False
            else:
                key_value = os.environ[key_name]
                if not key_value or len(key_value) < MIN_KEY_LENGTH:
                    messages.append(f"❌ API_KEY: {key_name} invalid")
                    all_valid = False
                else:
                    masked_value = key_value[:8] + "..."
                    messages.append(f"✅ API_KEY: {key_name} → {masked_value}")
        
        return all_valid, messages
    
    def execute_security_validation(self) -> bool:
        """
        Execute comprehensive security validation
        
        Returns:
            True if all security checks pass, False otherwise
        """
        print("🛡️  SEEKRATES_AI SECURITY GUARD - AGENT PROTECTION SYSTEM")
        print("📍 GPS Coordinate: fr_06_uc_01_ec_01_tc_301")
        print("🔐 Multi-Layer Security Validation")
        print("=" * 80)
        
        all_checks_passed = True
        
        # 1. AWS Identity Verification
        print("\n🔍 LAYER 1: AWS Identity Verification")
        aws_valid, aws_message = self.verify_aws_identity()
        print(aws_message)
        if not aws_valid:
            all_checks_passed = False
            self.security_failures.append("AWS_IDENTITY")
        
        # 2. Fernet Key Fingerprint Verification
        print("\n🔍 LAYER 2: Fernet Key Fingerprint Verification")
        fernet_valid, fernet_message = self.verify_fernet_fingerprint()
        print(fernet_message)
        if not fernet_valid:
            all_checks_passed = False
            self.security_failures.append("FERNET_FINGERPRINT")
        
        # 3. Deployment Environment Verification
        print("\n🔍 LAYER 3: Deployment Environment Verification")
        deployment_valid, deployment_message = self.verify_deployment_fingerprint()
        print(deployment_message)
        if not deployment_valid:
            all_checks_passed = False
            self.security_failures.append("DEPLOYMENT_FINGERPRINT")
        
        # 4. API Keys Validation
        print("\n🔍 LAYER 4: LLM API Keys Validation")
        api_keys_valid, api_messages = self.validate_api_keys()
        for message in api_messages:
            print(message)
        if not api_keys_valid:
            all_checks_passed = False
            self.security_failures.append("API_KEYS")
        
        return all_checks_passed
    
    def print_security_summary(self, all_valid: bool) -> None:
        """
        Print comprehensive security summary
        
        Args:
            all_valid: Whether all security validations passed
        """
        print("\n" + "=" * 80)
        print("🛡️  SECURITY VALIDATION SUMMARY")
        print("=" * 80)
        
        if all_valid:
            print("🎉 ALL SECURITY LAYERS VALIDATED")
            print("🚀 SEEKRATES_AI AGENT SYSTEM AUTHORIZED")
            print("💰 Monetizable agent protection: ACTIVE")
            print("🔒 Multi-layer security: PASSED")
            print("✅ Ready for production operations")
        else:
            print("💥 SECURITY VALIDATION FAILED")
            print("🚫 UNAUTHORIZED DEPLOYMENT DETECTED")
            print(f"❌ Failed security layers: {', '.join(self.security_failures)}")
            print("\n🛑 POSSIBLE SECURITY VIOLATIONS:")
            print("   • Running in unauthorized AWS account")
            print("   • Missing or invalid Fernet key fingerprint")
            print("   • Unauthorized deployment environment")
            print("   • Missing encrypted LLM API keys")
            print("\n💡 This system is protected against unauthorized use")
            print("🔐 Contact system owner for authorized deployment")
            print("\n🛑 EXECUTION TERMINATED FOR SECURITY")


def generate_initial_fingerprints():
    """
    Helper function to generate fingerprints for authorized environment setup
    Call this once in your authorized environment to get the fingerprints
    """
    print("🔧 FINGERPRINT GENERATION MODE")
    print("=" * 50)
    
    load_dotenv()
    guard = SecureRuntimeGuard()
    
    # Generate Fernet fingerprint
    fernet_fingerprint = guard.generate_fernet_fingerprint()
    if fernet_fingerprint:
        print(f"🔑 Fernet Fingerprint: {fernet_fingerprint}")
    else:
        print("❌ Could not generate Fernet fingerprint (FERNET_KEY missing?)")
    
    # Generate deployment fingerprint
    deployment_fingerprint = guard.generate_deployment_fingerprint()
    print(f"🖥️  Deployment Fingerprint: {deployment_fingerprint}")
    
    print("\n💡 Add these to AUTHORIZED_*_FINGERPRINTS lists in the script")


def main():
    """
    Main execution function - comprehensive security gate
    """
    # Check for fingerprint generation mode
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-fingerprints":
        generate_initial_fingerprints()
        return 0
    
    print("🛡️  Initializing SEEKRATES_AI Security Guard...")
    
    # Create guard instance
    guard = SecureRuntimeGuard()
    
    # Execute comprehensive security validation
    all_valid = guard.execute_security_validation()
    
    # Print summary
    guard.print_security_summary(all_valid)
    
    # Immediate halt on security failure
    if not all_valid:
        print(f"\n🚨 SECURITY GUARD: Terminating execution (Code: 1)")
        sys.exit(1)
    
    print("\n✅ SECURITY GUARD: All validations passed - agent authorized")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# =============================================================================
# SCRIPT DNA METADATA CONTINUED
# =============================================================================

# RELATED ARTIFACTS
related_hlr: str = "HLR_006 - Agent Protection Security"
related_fr: str = "FR_301 - Multi-Layer Security Guard"
related_uc: str = "UC_301 - Prevent Unauthorized Agent Deployment"
related_tc: str = "TC_301 - Security Layer Validation"
test_coverage_status: str = "Full"

# CANONIZATION PROTOCOL
canonization_status: str = "Production"
canonization_date: str = "2025-07-22"
canonization_authority: str = "Mohan Iyer - Business Owner"
change_authorization_required: bool = True

# CHANGE DOCUMENTATION
change_history = [
    {
        "change_date": "2025-07-22",
        "modified_by": "Claude",
        "reason": "Initial implementation of multi-layer agent protection system",
        "impact": "Prevents unauthorized deployment of monetizable AI agent in foreign environments",
        "validation": "Implements AWS identity verification, Fernet key fingerprinting, deployment environment validation, and API key checks",
        "affected_components": ["seekrates_ai protection", "Monetization security", "Agent authorization pipeline"]
    }
]

# OPERATIONAL NOTES
operational_notes: str = "Must be run before agent initialization. First run with --generate-fingerprints to capture authorized environment. Update TRUSTED_AWS_ACCOUNT_ID and fingerprint lists for your environment."
security_considerations: str = "Implements 4-layer protection: AWS identity, Fernet fingerprint, deployment environment, API keys. Prevents cloning and unauthorized deployment."
performance_considerations: str = "Single AWS STS call, local cryptographic operations, typical runtime <2 seconds. Fail-fast on any security violation."
object_design_notes: str = "Comprehensive security validation with clear separation of concerns. Each security layer is independently validated with detailed error reporting."