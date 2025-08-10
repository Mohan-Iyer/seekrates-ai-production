#!/usr/bin/env python3
"""
# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT  
# =============================================================================
# project_name: "decision_referee"
# module_name: "Validated Agent Dispatcher with Consensus Engine"
# script_id: "fr_04_uc_359_ec_02_tc_359"
# gps_coordinate: "fr_04_uc_359_ec_02_tc_359"
# script_name: "run_validated_agent.py"
# purpose: "Real LLM agent dispatcher with consensus engine integration"
# version: "4.0.0"
# status: "Production"
# author: "Claude"
# created_on: "2025-01-03T14:30:00Z"
# business_owner: "Mohan Iyer mohanpixels.net.nz"
# coding_engineer: "Claude"
# supervisor: "Yang - ChatGPT"
# =============================================================================
"""

# python src/agents/run_validated_agent.py claude "Explain quantum computing"
# python src/agents/run_validated_agent.py consensus "What is artificial intelligence?"
# python src/agents/run_validated_agent.py mistral "Write a Python function"

import time
import sys
import os
import subprocess
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import log_prompt_response

# ✅ TRIGGER CONSENSUS ENGINE - Import consensus functionality
try:
    from src.agents.consensus_engine import get_multi_agent_consensus, ConsensusEngine
    CONSENSUS_AVAILABLE = True
    print("✅ Consensus engine imported successfully")
except ImportError as e:
    CONSENSUS_AVAILABLE = False
    print(f"❌ Failed to import consensus engine: {e}")

try:
    from src.utils.secrets_manager import SecretsManager
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False

def load_env_file():
    """Load environment variables from .env file (LLM_LAW_3 compliant)."""
    env_file_path = Path(__file__).parent.parent.parent / '.env'
    if env_file_path.exists():
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def decrypt_and_inject_api_keys():
    """Decrypt and inject API keys from encrypted environment variables."""
    if not SECRETS_AVAILABLE:
        print('⚠️  SecretsManager not available - using plain keys if available')
        return
    
    try:
        secrets_manager = SecretsManager()
        
        # Map encrypted keys to plain keys
        key_mappings = {
            'OPENAI_API_KEY_ENC': 'OPENAI_API_KEY',
            'CLAUDE_API_KEY_ENC': 'ANTHROPIC_API_KEY', 
            'MISTRAL_API_KEY_ENC': 'MISTRAL_API_KEY',
            'GEMINI_API_KEY_ENC': 'GOOGLE_API_KEY'
        }
        
        for enc_key, plain_key in key_mappings.items():
            encrypted_value = os.getenv(enc_key)
            if encrypted_value:
                try:
                    decrypted = secrets_manager.decrypt(encrypted_value)
                    if decrypted:
                        os.environ[plain_key] = decrypted
                        print(f'🔓 Decrypted: {plain_key}')
                except Exception as e:
                    print(f'⚠️  Failed to decrypt {enc_key}: {e}')
    except Exception as e:
        print(f'⚠️  Decryption setup failed: {e}')

def load_api_key(provider):
    """
    Load API key from environment variables.
    
    Args:
        provider (str): The LLM provider name
        
    Returns:
        str: API key if found, None otherwise
        
    GPS: fr_04_uc_359_ec_02_tc_359_fn_001
    """
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',  # ← FIX: Was 'CLAUDE_API_KEY'
        'mistral': 'MISTRAL_API_KEY',
        'google': 'GOOGLE_API_KEY'
    }
    
    env_key = key_mapping.get(provider.lower())
    if env_key:
        api_key = os.getenv(env_key)
        if api_key:
            print(f"✅ Found {env_key}")
            return api_key
        else:
            print(f"❌ {env_key} not found in environment")
            return None
    return None

def call_openai_api(prompt, model="gpt-4"):
    """Call OpenAI API using curl (LLM_LAW_3 compliant)."""
    api_key = load_api_key('openai')
    if not api_key:
        return "❌ Error: OPENAI_API_KEY not found in environment"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    curl_command = [
        "curl", "-s", "-X", "POST",
        "https://api.openai.com/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(payload)
    ]
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                error = response_data.get('error', {})
                return f"❌ OpenAI API Error: {error.get('message', 'Unknown error')}"
        else:
            return f"❌ Curl Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ OpenAI API Timeout"
    except json.JSONDecodeError:
        return f"❌ Invalid JSON response from OpenAI API: {result.stdout[:200]}"
    except Exception as e:
        return f"❌ OpenAI API Exception: {str(e)}"

def call_anthropic_api(prompt, model="claude-3-5-sonnet-20241022"):
    """Call Anthropic Claude API using curl (LLM_LAW_3 compliant)."""
    api_key = load_api_key('anthropic')
    if not api_key:
        return "❌ Error: ANTHROPIC_API_KEY not found in environment"
    
    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    curl_command = [
        "curl", "-s", "-X", "POST",
        "https://api.anthropic.com/v1/messages",
        "-H", "Content-Type: application/json",
        "-H", f"x-api-key: {api_key}",
        "-H", "anthropic-version: 2023-06-01",
        "-d", json.dumps(payload)
    ]
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            if 'content' in response_data and len(response_data['content']) > 0:
                return response_data['content'][0]['text']
            else:
                error = response_data.get('error', {})
                return f"❌ Anthropic API Error: {error.get('message', 'Unknown error')}"
        else:
            return f"❌ Curl Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Anthropic API Timeout"
    except json.JSONDecodeError:
        return f"❌ Invalid JSON response from Anthropic API: {result.stdout[:200]}"
    except Exception as e:
        return f"❌ Anthropic API Exception: {str(e)}"

def call_mistral_api(prompt, model="mistral-large-latest"):
    """Call Mistral API using curl (LLM_LAW_3 compliant)."""
    api_key = load_api_key('mistral')
    if not api_key:
        return "❌ Error: MISTRAL_API_KEY not found in environment"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    curl_command = [
        "curl", "-s", "-X", "POST",
        "https://api.mistral.ai/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(payload)
    ]
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                error = response_data.get('error', {})
                return f"❌ Mistral API Error: {error.get('message', 'Unknown error')}"
        else:
            return f"❌ Curl Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Mistral API Timeout"
    except json.JSONDecodeError:
        return f"❌ Invalid JSON response from Mistral API: {result.stdout[:200]}"
    except Exception as e:
        return f"❌ Mistral API Exception: {str(e)}"

def call_ollama_api(prompt, model="llama3:8b"):
    """Call Ollama local API using curl (LLM_LAW_3 compliant)."""
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    curl_command = [
        "curl", "-s", "-X", "POST",
        f"{ollama_url}/api/generate",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            if 'response' in response_data:
                return response_data['response']
            else:
                return f"❌ Ollama API Error: {response_data.get('error', 'Unknown error')}"
        else:
            return f"❌ Curl Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Ollama API Timeout"
    except json.JSONDecodeError:
        return f"❌ Invalid JSON response from Ollama API: {result.stdout[:200]}"
    except Exception as e:
        return f"❌ Ollama API Exception: {str(e)}"

def run_consensus_mode(prompt):
    """
    ✅ TRIGGER CONSENSUS ENGINE
    Run consensus engine with multiple agents
    
    Args:
        prompt (str): The prompt to send to all agents
        
    Returns:
        dict: Consensus result
    """
    if not CONSENSUS_AVAILABLE:
        return {
            "success": False,
            "error": "Consensus engine not available",
            "results": []
        }
    
    try:
        print(f"🎯 Running CONSENSUS MODE")
        print(f"📝 Prompt: {prompt}")
        print("=" * 60)
        
        # Call consensus engine
        result = get_multi_agent_consensus(prompt)
        
        print(f"\n📊 CONSENSUS RESULTS:")
        print(f"✅ Success: {result.get('success', False)}")
        print(f"📈 Confidence: {result.get('confidence_score', 0)}")
        print(f"👥 Agents: {result.get('metadata', {}).get('successful_agents', 0)}/{result.get('metadata', {}).get('total_agents', 0)}")
        
        # ✅ LOG FAILURES CLEARLY
        print(f"\n🔍 AGENT STATUS BREAKDOWN:")
        failed_agents = []
        successful_agents = []
        
        for r in result.get("results", []):
            if not r.get("success", False):
                failed_agents.append(r)
                print(f"❌ {r.get('provider', 'unknown')} failed: {r.get('error', 'unknown error')}")
            else:
                successful_agents.append(r)
                print(f"✅ {r.get('provider', 'unknown')} succeeded: {r.get('confidence', 0.0):.2f} confidence")
        
        if result.get("consensus"):
            print(f"\n🤝 FINAL CONSENSUS:")
            print("-" * 40)
            print(result["consensus"])
            print("-" * 40)
        else:
            print(f"\n⚠️  NO CONSENSUS REACHED")
            print(f"Reason: {result.get('voting_details', {}).get('reason', 'unknown')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Consensus engine failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }
def debug_api_keys():
    """Debug function to check which API keys are available"""
    print("\n🔍 API KEY DEBUG:")
    print("=" * 40)
    
    keys_to_check = [
        'ANTHROPIC_API_KEY',
        'OPENAI_API_KEY', 
        'MISTRAL_API_KEY',
        'GEMINI_API_KEY'
    ]
    
    for key in keys_to_check:
        value = os.getenv(key)
        if value:
            # Show first 8 chars for security
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {key}: {masked}")
        else:
            print(f"❌ {key}: Not found")
    
    print("=" * 40)
def run_validated_agent(agent_name, prompt, model_config=None):
    """
    Run a validated agent with real LLM API calls and prompt logging.
    
    Args:
        agent_name (str): Name of the agent/provider to use
        prompt (str): The user prompt to process
        model_config (dict): Optional model configuration
        
    Returns:
        str: Agent response or error message
        
    GPS: fr_04_uc_359_ec_02_tc_359_fn_006
    """
    start_time = time.time()
    
    try:
        print(f"🤖 Running validated agent: {agent_name}")
        print(f"📝 Prompt: {prompt[:100]}...")
        
        # REAL API CALLS - No more stubs!
        if agent_name.lower() in ["gpt4", "gpt4_agent", "openai"]:
            response = call_openai_api(prompt)
        elif agent_name.lower() in ["claude", "claude_agent", "anthropic"]:
            response = call_anthropic_api(prompt)
        elif agent_name.lower() in ["mistral", "mistral_agent"]:
            response = call_mistral_api(prompt)
        elif agent_name.lower() in ["ollama", "ollama_agent"]:
            response = call_ollama_api(prompt)
        else:
            response = f"❌ Unknown agent: {agent_name}. Supported: gpt4, claude, mistral, ollama, consensus"
        
        # Calculate execution time for logging
        end_time = time.time()
        execution_time_ms = round((end_time - start_time) * 1000)
        
        # Log the prompt and response
        log_prompt_response(
            agent_name=agent_name,
            prompt=prompt,
            response=response,
            consensus=None,
            temperature=0.0,
            execution_time_ms=execution_time_ms,
            email=None
        )
        
        print(f"✅ Agent {agent_name} completed in {execution_time_ms}ms")
        return response
        
    except Exception as e:
        end_time = time.time()
        execution_time_ms = round((end_time - start_time) * 1000)
        
        error_response = f"❌ Error in {agent_name}: {str(e)}"
        
        log_prompt_response(
            agent_name=agent_name,
            prompt=prompt,
            response=error_response,
            consensus=None,
            temperature=0.0,
            execution_time_ms=execution_time_ms,
            email=None
        )
        
        print(f"❌ Agent {agent_name} failed after {execution_time_ms}ms: {str(e)}")
        return error_response

def validate_agent_response(response, validation_criteria=None):
    """Validate agent response against criteria."""
    if not response or len(response.strip()) == 0:
        return False
    
    # Check for error responses
    if response.startswith("❌"):
        return False
    
    return True

def main():
    """
    Main function - uses command line arguments or shows usage.
    
    GPS: fr_04_uc_359_ec_02_tc_359_fn_main
    """
    # Load environment variables from .env file
    load_env_file()
    
    # Decrypt and inject API keys
    decrypt_and_inject_api_keys()
    
    # Debug API keys
    debug_api_keys()
    
    if len(sys.argv) < 3:
        print("🚀 Real LLM Agent Dispatcher with Consensus Engine - Production Mode")
        print("=" * 70)
        print("Usage: python run_validated_agent.py <agent_name> <prompt>")
        print("\nSupported Agents:")
        print("  gpt4      - OpenAI GPT-4 (requires OPENAI_API_KEY)")
        print("  claude    - Anthropic Claude (requires ANTHROPIC_API_KEY)")
        print("  mistral   - Mistral AI (requires MISTRAL_API_KEY)")
        print("  ollama    - Local Ollama (requires Ollama server)")
        print("  consensus - Multi-agent consensus (requires multiple API keys)")
        print("\nExamples:")
        print("  python run_validated_agent.py claude 'What is the capital of France?'")
        print("  python run_validated_agent.py consensus 'What is artificial intelligence?'")
        print("  python run_validated_agent.py gpt4 'Explain quantum computing'")
        print("  python run_validated_agent.py mistral 'Write a Python function'")
        print("\nGPS Coordinate: fr_04_uc_359_ec_02_tc_359")
        print("LLM_LAW_3 Compliant: ✅ Curl-only HTTP requests")
        print("Consensus Engine: ✅ Multi-agent voting integration")
        return
    
    # Production mode - use command line arguments
    agent_name = sys.argv[1]
    prompt = " ".join(sys.argv[2:])
    
    print(f"🚀 Decision Referee - Agent Dispatcher v4.0")
    print(f"🤖 Agent: {agent_name}")
    print(f"📍 GPS: fr_04_uc_359_ec_02_tc_359")
    print("=" * 60)
    
    # Handle consensus mode specially
    if agent_name.lower() == "consensus":
        result = run_consensus_mode(prompt)
        
        print("\n🧪 FINAL RESULT:")
        print(json.dumps(result, indent=2))
        
        # ✅ EXIT WITH STATUS - If consensus is not achieved
        if not result.get("success", False):
            print(f"\n❌ CONSENSUS FAILED")
            print(f"Reason: {result.get('error', 'unknown')}")
            
            # ✅ LOG FAILURES CLEARLY - Show which agents failed
            failed_count = 0
            for r in result.get("results", []):
                if not r.get("success", False):
                    failed_count += 1
                    print(f"❌ {r.get('provider', 'unknown')} failed: {r.get('error', 'unknown error')}")
            
            print(f"\n💔 {failed_count} agent(s) failed. Check logs for details.")
            sys.exit("Consensus failed. Check logs.")
        else:
            print(f"\n✅ CONSENSUS ACHIEVED")
            print(f"Confidence: {result.get('confidence_score', 0)}%")
            print(f"Success rate: {result.get('metadata', {}).get('successful_agents', 0)}/{result.get('metadata', {}).get('total_agents', 0)}")
    else:
        # Single agent mode
        response = run_validated_agent(agent_name, prompt)
        is_valid = validate_agent_response(response)
        
        print(f"\n📊 Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print(f"✅ Validation Status: {'VALID' if is_valid else 'INVALID'}")
        
        if not is_valid:
            sys.exit("Agent response validation failed.")
    
    print(f"🏁 Execution Complete")

if __name__ == "__main__":
    main()