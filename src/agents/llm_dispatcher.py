#!/usr/bin/env python3
# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "Agent Core Infrastructure"
# script_id: "fr_04_uc_359_ec_03_tc_360"
# gps_coordinate: "fr_04_uc_359_ec_03_tc_360"
# script_name: "src/agents/llm_dispatcher.py"
# purpose: "Route prompts to specific LLM agents with real API integration"
# version: "2.0.0"
# status: "Production - RESPONSE FORMAT COMPLIANT"
# =============================================================================

#!/usr/bin/env python3
# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# ... existing DNA metadata ...

import json
import logging
import subprocess
import sys
import traceback
import os  # ← CRITICAL: Must be at global scope
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()

# ============================================================================
# CANONICAL PATH RESOLUTION - DIRECTORY_MAP.YAML AUTHORITY
# ============================================================================
try:
    with open("directory_map.yaml", "r") as f:
        directory_map = yaml.safe_load(f)
    
    # Resolve canonical paths using correct keys
    project_root = Path(directory_map["project_root"])
    src_root = Path(directory_map["src"])  # or directory_map["source_root"]
    utils_dir = Path(directory_map["utilities"]["utils_root"])
    agents_dir = Path(directory_map["agents"]["agents_root"])
    
    # Add src to path only if needed
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    
    logging.info(f"✅ Canonical paths loaded from directory_map.yaml")
        
except (FileNotFoundError, KeyError, yaml.YAMLError) as e:  # ← EXPANDED ERROR HANDLING
    # Fallback for development
    logging.warning(f"directory_map.yaml issue ({e}) - using fallback paths")
    project_root = Path(".")
    src_root = Path("src")
    utils_dir = Path("src/utils") 
    agents_dir = Path("src/agents")
    
    # Add fallback path to sys.path  ← MISSING IN YOUR VERSION
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))        
except FileNotFoundError:
    # Fallback for development
    logging.warning("directory_map.yaml not found - using fallback paths")
    project_root = Path(".")
    src_root = Path("src")
    utils_dir = Path("src/utils") 
    agents_dir = Path("src/agents")

# Now import using resolved paths
try:
    from src.utils.secrets_manager import SecretsManager
    from src.utils.logging_utils import log_prompt_response
except ImportError as e:
    print(f"Import error with canonical paths: {e}")
    logging.basicConfig(level=logging.INFO)

# Setup logging
logger = logging.getLogger(__name__)

class LLMDispatcher:
    """Central dispatcher for routing prompts to specific LLM agents"""
    
    def __init__(self):
        """Initialize the LLM dispatcher with secrets manager"""
        try:
            self.secrets_manager = SecretsManager()
        except Exception as e:
            logger.warning(f"SecretsManager init failed: {e}")
            self.secrets_manager = None
            
        self.supported_agents = {
            "claude": self._handle_claude,
            "openai": self._handle_openai, 
            "gpt4": self._handle_openai,
            "gpt": self._handle_openai,
            "chatgpt": self._handle_openai,
            "mistral": self._handle_mistral,
            "gemini": self._handle_gemini,
            "google": self._handle_gemini,
            "bard": self._handle_gemini,
            "ollama": self._handle_ollama,
            "llama": self._handle_ollama, 
            "ai21": self._handle_ai21,
            "jurassic": self._handle_ai21,
            "cohere": self._handle_cohere,
            "command": self._handle_cohere
            
        }
        logger.info("LLM Dispatcher initialized with supported agents: %s", list(self.supported_agents.keys()))
    
    def _build_standardized_response(
        self, 
        answer: str, 
        agent_name: str,
        success: bool = True,
        model: str = None,
        usage: Dict = None,
        finish_reason: str = None,
        confidence: float = 0.9,
        error: str = None
    ) -> Dict[str, Any]:
        """Build standardized response format for all agents"""
        
        # Normalize usage across providers
        normalized_usage = {}
        if usage:
            # Handle different provider usage formats
            if "input_tokens" in usage and "output_tokens" in usage:
                # Anthropic format
                normalized_usage = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                }
            elif "prompt_tokens" in usage and "completion_tokens" in usage:
                # OpenAI/Mistral format
                normalized_usage = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
            elif "promptTokenCount" in usage and "candidatesTokenCount" in usage:
                # Gemini format
                normalized_usage = {
                    "input_tokens": usage.get("promptTokenCount", 0),
                    "output_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0)
                }
            
            # Add service tier if available
            if "service_tier" in usage:
                normalized_usage["service_tier"] = usage["service_tier"]
        
        response = {
            "answer": answer,
            "confidence": confidence,
            "success": success,
            "agent_name": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if model:
            response["model"] = model
        if normalized_usage:
            response["usage"] = normalized_usage
        if finish_reason:
            response["finish_reason"] = finish_reason
        if error:
            response["error"] = error
            
        return response
    
    def _get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key with fallback to multiple sources and proper logging.
        Uses global os import - no local imports to avoid scope conflicts.
        """
        
        # Map service names to environment variable names
        service_mapping = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY", 
            "mistral": "MISTRAL_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "ai21": "AI21_API_KEY",
            "cohere": "COHERE_API_KEY"
        }
        
        env_var_name = service_mapping.get(service.lower())
        if not env_var_name:
            logger.warning(f"⚠️  Unknown service '{service}' - no API key mapping")
            return None
        
        # Try SecretsManager first (if available)
        if self.secrets_manager:
            try:
                # Try encrypted environment variable
                encrypted_env_var = f"{service.upper()}_API_KEY_ENC"
                encrypted_key = os.getenv(encrypted_env_var)  # ← Uses global os
                if encrypted_key:
                    decrypted = self.secrets_manager.decrypt(encrypted_key)
                    if decrypted:
                        logger.info(f"✅ {env_var_name} loaded from encrypted source")
                        return decrypted
            except Exception as e:
                logger.warning(f"SecretsManager decryption failed: {e}")
        
        # CRITICAL FALLBACK: Direct environment variable access
        plain_key = os.getenv(env_var_name)  # ← Uses global os, no scope conflict
        if plain_key and plain_key.strip():
            logger.info(f"✅ {env_var_name} loaded from .env")
            return plain_key.strip()
        
        # Key not found - log warning
        logger.warning(f"⚠️  {env_var_name} is missing or empty")
        return None

    def dispatch(self, agent: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Route prompt to specified agent and return standardized response
        
        Args:
            agent: The agent to use (claude, openai, mistral, gemini, ollama)
            prompt: The prompt to send
            **kwargs: Additional arguments for the specific agent
            
        Returns:
            Dict with standardized response format
        """
        if agent.lower() not in self.supported_agents:
            return self._build_standardized_response(
                answer=f"Unsupported agent: {agent}",
                agent_name=agent,
                success=False,
                confidence=0.0,
                error="unsupported_agent"
            )
        
        try:
            handler = self.supported_agents[agent.lower()]
            return handler(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error dispatching to {agent}: {e}")
            return self._build_standardized_response(
                answer=f"Error with {agent}: {str(e)}",
                agent_name=agent,
                success=False,
                confidence=0.0,
                error="dispatch_error"
            )

    def _handle_claude(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Claude API requests"""
        try:
            api_key = self._get_api_key("anthropic")
            if not api_key:
                return self._build_standardized_response(
                    answer="Claude API key not configured",
                    agent_name="claude",
                    success=False,
                    confidence=0.0,
                    error="missing_api_key"
                )
            
            # Use curl to call Claude API
            curl_command = [
                "curl", "-X", "POST",
                "https://api.anthropic.com/v1/messages",
                "-H", "Content-Type: application/json",
                "-H", f"x-api-key: {api_key}",
                "-H", "anthropic-version: 2023-06-01",
                "-d", json.dumps({
                    "model": kwargs.get("model", "claude-3-5-sonnet-20241022"),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "messages": [{"role": "user", "content": prompt}]
                }),
                "--max-time", "30"
            ]
            
            result = subprocess.run(curl_command, capture_output=True, text=True, timeout=35)
            
            if result.returncode != 0:
                return self._build_standardized_response(
                    answer=f"Claude API error: {result.stderr}",
                    agent_name="claude",
                    success=False,
                    confidence=0.0,
                    error="api_error"
                )
            
            response_data = json.loads(result.stdout)
            
            if "error" in response_data:
                return self._build_standardized_response(
                    answer=f"Claude API error: {response_data['error'].get('message', 'Unknown error')}",
                    agent_name="claude",
                    success=False,
                    confidence=0.0,
                    error="api_error"
                )
            
            response_text = response_data["content"][0]["text"]
            finish_reason = response_data.get("stop_reason")
            
            return self._build_standardized_response(
                answer=response_text,
                agent_name="claude",
                model=response_data.get("model", "claude-3-sonnet-20240229"),
                usage=response_data.get("usage", {}),
                finish_reason=finish_reason,
                confidence=0.9,
                success=True
            )
            
        except Exception as e:
            # ═══════════════════════════════════════════════════════════════
            # 🚨 ENHANCED CLAUDE API DEBUGGING - COMPREHENSIVE ERROR ANALYSIS
            # ═══════════════════════════════════════════════════════════════
            print(f"\n❌ CLAUDE API ERROR - COMPREHENSIVE DEBUG ANALYSIS")
            print(f"=" * 70)
            print(f"🔍 Error Type: {type(e).__name__}")
            print(f"🔍 Error Message: {str(e)}")
            print(f"🔍 Error Args: {e.args}")
            
            # API Key Analysis
            api_key = self._get_api_key("anthropic")
            print(f"🔑 API Key Present: {bool(api_key)}")
            if api_key:
                print(f"🔑 API Key Length: {len(api_key)} chars")
                print(f"🔑 API Key Prefix: {api_key[:15]}...")
                print(f"🔑 API Key Type: {type(api_key).__name__}")
            else:
                print(f"🔑 API Key: COMPLETELY MISSING")
                
            # Environment Variable Analysis  
            env_key = os.getenv("ANTHROPIC_API_KEY")
            enc_key = os.getenv("ANTHROPIC_API_KEY_ENC")
            print(f"🌍 ANTHROPIC_API_KEY env: {bool(env_key)} ({len(env_key) if env_key else 0} chars)")
            print(f"🌍 ANTHROPIC_API_KEY_ENC env: {bool(enc_key)} ({len(enc_key) if enc_key else 0} chars)")
            
            # Prompt Analysis
            print(f"📝 Prompt Length: {len(prompt)} chars")
            print(f"📝 Prompt Preview: {prompt[:100]}...")
            
            # Curl Command Analysis (if curl-related error)
            if "curl" in str(e).lower() or "subprocess" in str(e).lower():
                print(f"🔨 CURL/SUBPROCESS ERROR DETECTED")
                print(f"🔨 This suggests a system-level issue with curl command")
                
            # Full traceback
            print(f"📚 FULL TRACEBACK:")
            import traceback
            traceback.print_exc()
            print(f"=" * 70)
            
            # Log to file for persistence
            logger.error(f"Claude handler error: {str(e)}")
            logger.error(f"Claude API key present: {bool(api_key)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return self._build_standardized_response(
                answer=f"Claude processing error: {str(e)}",
                agent_name="claude",
                success=False,
                confidence=0.0,
                error="processing_error"
            )
            
    
    def _handle_openai(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle OpenAI API requests"""
        try:
            api_key = self._get_api_key("openai")
            if not api_key:
                return self._build_standardized_response(
                    answer="OpenAI API key not configured",
                    agent_name="openai",
                    success=False,
                    confidence=0.0,
                    error="missing_api_key"
                )
            
            # Use curl to call OpenAI API
            curl_command = [
                "curl", "-X", "POST",
                "https://api.openai.com/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", json.dumps({
                    "model": kwargs.get("model", "gpt-4o-mini"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }),
                "--max-time", "30"
            ]
            
            result = subprocess.run(curl_command, capture_output=True, text=True, timeout=35)
            
            if result.returncode != 0:
                return self._build_standardized_response(
                    answer=f"OpenAI API error: {result.stderr}",
                    agent_name="openai",
                    success=False,
                    confidence=0.0,
                    error="api_error"
                )
            
            response_data = json.loads(result.stdout)
            
            if "error" in response_data:
                return self._build_standardized_response(
                    answer=f"OpenAI API error: {response_data['error'].get('message', 'Unknown error')}",
                    agent_name="openai",
                    success=False,
                    confidence=0.0,
                    error="api_error"
                )
            
            choice = response_data["choices"][0]
            response_text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            
            return self._build_standardized_response(
                answer=response_text,
                agent_name="openai",
                model=response_data.get("model", "gpt-4o-mini"),
                usage=response_data.get("usage", {}),
                finish_reason=finish_reason,
                confidence=0.9,
                success=True
            )
            
        except Exception as e:
            logger.error(f"OpenAI handler error: {str(e)}")
            return self._build_standardized_response(
                answer=f"OpenAI processing error: {str(e)}",
                agent_name="openai",
                success=False,
                confidence=0.0,
                error="processing_error"
            )
    
    def _handle_mistral(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Mistral AI API requests with enhanced debugging and retry logic"""
        try:
            api_key = self._get_api_key("mistral")
            if not api_key:
                error_log = {
                    "agent": "mistral",
                    "status": "error", 
                    "error": "API key not configured",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                print(f"❌ MISTRAL DEBUG: {json.dumps(error_log)}")
                return self._build_standardized_response(
                    answer="Mistral API key not configured",
                    agent_name="mistral",
                    success=False,
                    confidence=0.0,
                    error="missing_api_key"
                )
            
            print(f"🔍 MISTRAL DEBUG: Starting API call with {len(prompt)} char prompt")
            print(f"🔍 MISTRAL DEBUG: API key length: {len(api_key)} chars")
            
            # Enhanced curl command with longer timeout
            curl_command = [
                "curl", "-X", "POST",
                "https://api.mistral.ai/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", json.dumps({
                    "model": kwargs.get("model", "mistral-small-latest"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }),
                "--max-time", "45"  # ← INCREASED TIMEOUT
            ]
            
            # Retry logic
            for attempt in range(2):
                print(f"🔍 MISTRAL DEBUG: Attempt {attempt + 1}/2")
                
                result = subprocess.run(curl_command, capture_output=True, text=True, timeout=50)
                
                print(f"🔍 MISTRAL DEBUG: Return code: {result.returncode}")
                print(f"🔍 MISTRAL DEBUG: Response length: {len(result.stdout)} chars")
                
                if result.stderr:
                    print(f"🔍 MISTRAL DEBUG: Stderr: {result.stderr}")
                
                if result.returncode == 0:
                    break
                elif attempt == 0:
                    print(f"⚠️ MISTRAL DEBUG: Attempt 1 failed, retrying after 2s delay...")
                    import time
                    time.sleep(2)
                else:
                    error_log = {
                        "agent": "mistral",
                        "status": "error",
                        "error": f"Connection failed after 2 attempts: {result.stderr}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    print(f"❌ MISTRAL DEBUG: {json.dumps(error_log)}")
                    return self._build_standardized_response(
                        answer=f"Mistral API connection error: {result.stderr}",
                        agent_name="mistral",
                        success=False,
                        confidence=0.0,
                        error="connection_error"
                    )
            
            print(f"🔍 MISTRAL DEBUG: Raw response preview: {result.stdout[:200]}...")
            
            response_data = json.loads(result.stdout)
            
            if "error" in response_data:
                error_log = {
                    "agent": "mistral",
                    "status": "error",
                    "error": response_data['error'].get('message', 'Unknown API error'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                print(f"❌ MISTRAL DEBUG: {json.dumps(error_log)}")
                return self._build_standardized_response(
                    answer=f"Mistral API error: {response_data['error'].get('message', 'Unknown error')}",
                    agent_name="mistral",
                    success=False,
                    confidence=0.0,
                    error="api_error"
                )
            
            choice = response_data["choices"][0]
            response_text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            
            success_log = {
                "agent": "mistral",
                "status": "success",
                "response_length": len(response_text),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"✅ MISTRAL DEBUG: {json.dumps(success_log)}")
            
            return self._build_standardized_response(
                answer=response_text,
                agent_name="mistral",
                model=response_data.get("model", "mistral-small-latest"),
                usage=response_data.get("usage", {}),
                finish_reason=finish_reason,
                confidence=0.9,
                success=True
            )
            
        except json.JSONDecodeError as e:
            error_log = {
                "agent": "mistral",
                "status": "error",
                "error": f"JSON decode error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"❌ MISTRAL DEBUG: {json.dumps(error_log)}")
            return self._build_standardized_response(
                answer="Mistral returned invalid JSON response",
                agent_name="mistral",
                success=False,
                confidence=0.0,
                error=f"json_decode_error: {str(e)}"
            )
        except Exception as e:
            error_log = {
                "agent": "mistral",
                "status": "error",
                "error": f"Processing error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"❌ MISTRAL DEBUG: {json.dumps(error_log)}")
            import traceback
            traceback.print_exc()
            
            return self._build_standardized_response(
                answer=f"Mistral processing error: {str(e)}",
                agent_name="mistral",
                success=False,
                confidence=0.0,
                error="processing_error"
            )

    def _handle_gemini(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Google Gemini API requests with enhanced debugging and retry logic"""
        try:
            api_key = self._get_api_key("gemini")
            if not api_key:
                error_log = {
                    "agent": "gemini",
                    "status": "error",
                    "error": "API key not configured",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                print(f"❌ GEMINI DEBUG: {json.dumps(error_log)}")
                return self._build_standardized_response(
                    answer="Gemini API key not configured",
                    agent_name="gemini",
                    success=False,
                    confidence=0.0,
                    error="missing_api_key"
                )
            
            print(f"🔍 GEMINI DEBUG: Starting API call with {len(prompt)} char prompt")
            print(f"🔍 GEMINI DEBUG: API key length: {len(api_key)} chars")
            
            model = kwargs.get("model", "gemini-1.5-flash")
            
            curl_command = [
                "curl", "-X", "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "maxOutputTokens": kwargs.get("max_tokens", 1000),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                }),
                "--max-time", "45"  # ← INCREASED TIMEOUT
            ]
            
            # Retry logic with delay
            for attempt in range(2):
                print(f"🔍 GEMINI DEBUG: Attempt {attempt + 1}/2")
                
                result = subprocess.run(curl_command, capture_output=True, text=True, timeout=50)
                
                print(f"🔍 GEMINI DEBUG: Return code: {result.returncode}")
                print(f"🔍 GEMINI DEBUG: Response length: {len(result.stdout)} chars")
                
                if result.stderr:
                    print(f"🔍 GEMINI DEBUG: Stderr: {result.stderr}")
                
                if result.returncode == 0:
                    break
                elif attempt == 0:
                    print(f"⚠️ GEMINI DEBUG: Attempt 1 failed, retrying after 3s delay...")
                    import time
                    time.sleep(3)  # Longer delay for Gemini
                else:
                    error_log = {
                        "agent": "gemini",
                        "status": "error",
                        "error": f"Connection failed after 2 attempts: {result.stderr}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    print(f"❌ GEMINI DEBUG: {json.dumps(error_log)}")
                    return self._build_standardized_response(
                        answer=f"Gemini API connection error: {result.stderr}",
                        agent_name="gemini",
                        success=False,
                        confidence=0.0,
                        error="connection_error"
                    )
            
            print(f"🔍 GEMINI DEBUG: Raw response preview: {result.stdout[:200]}...")
            
            response_data = json.loads(result.stdout)
            
            if "error" in response_data:
                error_log = {
                    "agent": "gemini",
                    "status": "error",
                    "error": response_data['error'].get('message', 'Unknown API error'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                print(f"❌ GEMINI DEBUG: {json.dumps(error_log)}")
                return self._build_standardized_response(
                    answer=f"Gemini API error: {response_data['error'].get('message', 'Unknown error')}",
                    agent_name="gemini",
                    success=False,
                    confidence=0.0,
                    error="api_error"
                )
            
            if not response_data.get("candidates") or not response_data["candidates"][0].get("content"):
                error_log = {
                    "agent": "gemini",
                    "status": "error",
                    "error": "No content in response",
                    "response_structure": list(response_data.keys()),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                print(f"❌ GEMINI DEBUG: {json.dumps(error_log)}")
                return self._build_standardized_response(
                    answer="Gemini returned no content",
                    agent_name="gemini",
                    success=False,
                    confidence=0.0,
                    error="no_content"
                )
            
            candidate = response_data["candidates"][0]
            response_text = candidate["content"]["parts"][0]["text"]
            finish_reason = candidate.get("finishReason")
            
            success_log = {
                "agent": "gemini",
                "status": "success",
                "response_length": len(response_text),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"✅ GEMINI DEBUG: {json.dumps(success_log)}")
            
            return self._build_standardized_response(
                answer=response_text,
                agent_name="gemini",
                model=model,
                usage=response_data.get("usageMetadata", {}),
                finish_reason=finish_reason,
                confidence=0.9,
                success=True
            )
            
        except json.JSONDecodeError as e:
            error_log = {
                "agent": "gemini",
                "status": "error",
                "error": f"JSON decode error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"❌ GEMINI DEBUG: {json.dumps(error_log)}")
            return self._build_standardized_response(
                answer="Gemini returned invalid JSON response",
                agent_name="gemini",
                success=False,
                confidence=0.0,
                error=f"json_decode_error: {str(e)}"
            )
        except Exception as e:
            error_log = {
                "agent": "gemini",
                "status": "error",
                "error": f"Processing error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"❌ GEMINI DEBUG: {json.dumps(error_log)}")
            import traceback
            traceback.print_exc()
            
            return self._build_standardized_response(
                answer=f"Gemini processing error: {str(e)}",
                agent_name="gemini",
                success=False,
                confidence=0.0,
                error="processing_error"
            )

    def _handle_cohere(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Cohere API requests with enhanced debugging and retry logic"""
        try:
            api_key = self._get_api_key("cohere")
            if not api_key:
                error_log = {
                    "agent": "cohere",
                    "status": "error",
                    "error": "API key not configured",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                print(f"❌ COHERE DEBUG: {json.dumps(error_log)}")
                return self._build_standardized_response(
                    answer="Cohere API key not configured",
                    agent_name="cohere",
                    success=False,
                    confidence=0.0,
                    error="missing_api_key"
                )
            
            print(f"🔍 COHERE DEBUG: Starting API call with {len(prompt)} char prompt")
            
            model = kwargs.get("model", "command-r-plus")
            
            curl_command = [
                "curl", "-X", "POST",
                "https://api.cohere.ai/v1/chat",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", json.dumps({
                    "model": model,
                    "message": prompt,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }),
                "--max-time", "45"  # ← INCREASED TIMEOUT
            ]
            
            # Retry logic
            for attempt in range(2):
                print(f"🔍 COHERE DEBUG: Attempt {attempt + 1}/2")
                
                result = subprocess.run(curl_command, capture_output=True, text=True, timeout=50)
                
                print(f"🔍 COHERE DEBUG: Return code: {result.returncode}")
                
                if result.returncode == 0:
                    break
                elif attempt == 0:
                    print(f"⚠️ COHERE DEBUG: Attempt 1 failed, retrying after 2s delay...")
                    import time
                    time.sleep(2)
                else:
                    error_log = {
                        "agent": "cohere",
                        "status": "error",
                        "error": f"Connection failed after 2 attempts: {result.stderr}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    print(f"❌ COHERE DEBUG: {json.dumps(error_log)}")
                    return self._build_standardized_response(
                        answer=f"Cohere API connection error: {result.stderr}",
                        agent_name="cohere",
                        success=False,
                        confidence=0.0,
                        error="connection_error"
                    )
            
            response_data = json.loads(result.stdout)
            
            if "message" in response_data and "error" in response_data.get("message", "").lower():
                error_log = {
                    "agent": "cohere",
                    "status": "error",
                    "error": response_data['message'],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                print(f"❌ COHERE DEBUG: {json.dumps(error_log)}")
                return self._build_standardized_response(
                    answer=f"Cohere API error: {response_data['message']}",
                    agent_name="cohere",
                    success=False,
                    confidence=0.0,
                    error="api_error"
                )
            
            response_text = response_data.get("text", "No response from Cohere")
            finish_reason = response_data.get("finish_reason")
            
            success_log = {
                "agent": "cohere",
                "status": "success",
                "response_length": len(response_text),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"✅ COHERE DEBUG: {json.dumps(success_log)}")
            
            return self._build_standardized_response(
                answer=response_text,
                agent_name="cohere",
                model=model,
                usage=response_data.get("meta", {}).get("tokens", {}),
                finish_reason=finish_reason,
                confidence=0.9,
                success=True
            )
            
        except json.JSONDecodeError as e:
            error_log = {
                "agent": "cohere",
                "status": "error",
                "error": f"JSON decode error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"❌ COHERE DEBUG: {json.dumps(error_log)}")
            return self._build_standardized_response(
                answer="Cohere returned invalid JSON response",
                agent_name="cohere",
                success=False,
                confidence=0.0,
                error=f"json_decode_error: {str(e)}"
            )
        except Exception as e:
            error_log = {
                "agent": "cohere",
                "status": "error",
                "error": f"Processing error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            print(f"❌ COHERE DEBUG: {json.dumps(error_log)}")
            import traceback
            traceback.print_exc()
            
            return self._build_standardized_response(
                answer=f"Cohere processing error: {str(e)}",
                agent_name="cohere",
                success=False,
                confidence=0.0,
                error="processing_error"
            )

    def _handle_ollama(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Ollama local model requests"""
        try:
            model = kwargs.get("model", "llama3.2")
            
            # Use curl to call local Ollama API
            curl_command = [
                "curl", "-X", "POST",
                "http://localhost:11434/api/generate",
                "-H", "Content-Type: application/json", 
                "-d", json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                }),
                "--max-time", "60"
            ]
            
            result = subprocess.run(curl_command, capture_output=True, text=True, timeout=65)
            
            if result.returncode != 0:
                return self._build_standardized_response(
                    answer="Ollama connection failed - is Ollama server running on localhost:11434?",
                    agent_name="ollama",
                    success=False,
                    confidence=0.0,
                    error=f"ollama_connection_error: {result.stderr}"
                )
            
            response_data = json.loads(result.stdout)
            
            if "error" in response_data:
                return self._build_standardized_response(
                    answer=f"Ollama error: {response_data['error']}",
                    agent_name="ollama",
                    success=False,
                    confidence=0.0,
                    error="ollama_api_error"
                )
            
            response_text = response_data.get("response", "No response from Ollama")
            
            # Ollama provides some metadata
            context = response_data.get("context", [])
            done = response_data.get("done", False)
            
            # Create basic usage info since Ollama doesn't provide tokens
            estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            estimated_output_tokens = len(response_text.split()) * 1.3
            
            usage = {
                "input_tokens": int(estimated_input_tokens),
                "output_tokens": int(estimated_output_tokens),
                "total_tokens": int(estimated_input_tokens + estimated_output_tokens)
            }
            
            return self._build_standardized_response(
                answer=response_text,
                agent_name="ollama",
                model=model,
                usage=usage,
                finish_reason="stop" if done else "incomplete",
                confidence=0.8,
                success=True
            )
            
        except json.JSONDecodeError as e:
            return self._build_standardized_response(
                answer="Ollama returned invalid JSON response",
                agent_name="ollama",
                success=False,
                confidence=0.0,
                error=f"json_decode_error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Ollama handler error: {str(e)}")
            return self._build_standardized_response(
                answer=f"Ollama processing error: {str(e)}",
                agent_name="ollama",
                success=False,
                confidence=0.0,
                error="processing_error"
            )

    def _handle_ai21(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle AI21 API requests with correct endpoint"""
        try:
            api_key = self._get_api_key("ai21")
            if not api_key:
                return self._build_standardized_response(
                    answer="AI21 API key not configured",
                    agent_name="ai21",
                    success=False,
                    confidence=0.0,
                    error="missing_api_key"
                )
            
            model = kwargs.get("model", "jamba-1.5-mini")
            
            # ✅ CORRECT AI21 ENDPOINT
            curl_command = [
                "curl", "-X", "POST",
                f"https://api.ai21.com/studio/v1/{model}/complete",  # ← FIXED URL
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {api_key}",
                "-d", json.dumps({
                    "prompt": prompt,
                    "maxTokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }),
                "--max-time", "30"
            ]
            
            result = subprocess.run(curl_command, capture_output=True, text=True, timeout=35)
            
            if result.returncode != 0:
                return self._build_standardized_response(
                    answer=f"AI21 API connection failed: {result.stderr}",
                    agent_name="ai21",
                    success=False,
                    confidence=0.0,
                    error="connection_error"
                )
            
            response_data = json.loads(result.stdout)
            
            # AI21 completion format
            if "completions" in response_data and response_data["completions"]:
                response_text = response_data["completions"][0]["data"]["text"]
            elif "text" in response_data:
                response_text = response_data["text"]
            else:
                return self._build_standardized_response(
                    answer=f"AI21 parsing error. Response: {str(response_data)[:200]}",
                    agent_name="ai21",
                    success=False,
                    confidence=0.0,
                    error="parsing_error"
                )
            
            return self._build_standardized_response(
                answer=response_text,
                agent_name="ai21",
                model=model,
                confidence=0.9,
                success=True
            )
            
        except Exception as e:
            return self._build_standardized_response(
                answer=f"AI21 processing error: {str(e)}",
                agent_name="ai21",
                success=False,
                confidence=0.0,
                error="processing_error"
            )
                        
# Global dispatcher instance
dispatcher = LLMDispatcher()

def call_llm_agent(agent_name: str, prompt: str, timeout: Optional[int] = 60, **kwargs) -> Dict[str, Any]:
    """
    Public interface for dispatching prompts to LLM agents (backward compatibility)
    
    Args:
        agent_name: Agent name (openai, mistral, gemini, claude, ollama, ai21, cohere)
        prompt: User prompt text
        timeout: Timeout in seconds (ignored for now, kept for compatibility)
        **kwargs: Additional parameters like model, temperature, etc.
        
    Returns:
        Standardized response dictionary
    """
    return dispatcher.dispatch(agent_name, prompt, **kwargs)  # ✅ Pass through kwargs

def dispatch_prompt(agent: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Public interface for dispatching prompts to LLM agents
    
    Args:
        agent: Agent name (openai, mistral, gemini, claude, ollama)
        prompt: User prompt text
        **kwargs: Additional agent parameters
        
    Returns:
        Standardized response dictionary
    """
    return dispatcher.dispatch(agent, prompt, **kwargs)

def get_supported_agents() -> list:
    """Return list of supported agent names"""
    return list(dispatcher.supported_agents.keys())

if __name__ == "__main__":
    # Test the dispatcher
    test_prompt = "What is the capital of Germany?"
    
    print("Testing LLM Dispatcher...")
    print(f"Supported agents: {get_supported_agents()}")
    
    # Test each agent
    for agent in ["openai", "mistral", "gemini"]:
        print(f"\n--- Testing {agent} ---")
        result = call_llm_agent(agent, test_prompt)
        print(f"Success: {result.get('success', False)}")
        print(f"Response: {result.get('answer', 'No response')}")
        if result.get('error'):
            print(f"Error: {result['error']}")