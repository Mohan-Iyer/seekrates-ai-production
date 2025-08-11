# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "runtime"
# script_id: "fr_11_uc_091_ec_07_tc_250"
# gps_coordinate: "fr_11_uc_091_ec_07_tc_250"
# script_name: "app.py"
# purpose: "Run AWS secrets bootstrap before serving; toggle via BOOTSTRAP_ENABLED"
# version: "1.0.0"
# status: "Production"
# author: "MI"
# coding_engineer: "Claude"
# supervisor: "Yang - ChatGPT"
# business_owner: "Mohan Iyer mohan@pixels.net.nz"
# =============================================================================

import os

def run_bootstrap():
    """Run AWS secrets bootstrap if enabled."""
    if os.getenv("BOOTSTRAP_ENABLED", "true").lower() == "true":
        try:
            from src.secrets.inject_llm_keys_from_aws import load_secrets
            load_secrets()
            print("✅ AWS secrets bootstrap completed")
        except Exception as e:
            # Fail fast with a clear message (no secrets in logs)
            raise SystemExit(f"Bootstrap failed: {e}")
    else:
        print("ℹ️  Bootstrap disabled via BOOTSTRAP_ENABLED=false")

from flask import Flask, render_template, request, jsonify, send_from_directory
import sys
import yaml
from pathlib import Path

# Add paths for imports
sys.path.append('src/agents')
sys.path.append('src/core')

from consensus_engine import generate_expert_panel_response_v3

app = Flask(__name__, static_folder='static', template_folder='static')

# =============================================================================
# PATCH 01A: PROVIDER NORMALIZATION HELPERS
# =============================================================================
_ENV_MAP = {
    "openai":  ["OPENAI_API_KEY"],
    "claude":  ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "cohere":  ["COHERE_API_KEY"],
    "gemini":  ["GEMINI_API_KEY", "GOOGLE_API_KEY"],  # alias
    "together":["TOGETHER_API_KEY"],
}
_ALIAS = {
    "gpt4": "openai", "gpt-4": "openai",
    "anthropic": "claude", "claude-3": "claude",
    "google": "gemini", "gemini-pro": "gemini", "vertex-gemini": "gemini",
}

def _available_providers_from_env():
    return {p for p, names in _ENV_MAP.items() if any(os.getenv(n) for n in names)}

def _normalize_providers_from_payload(data: dict):
    # accept both keys
    providers = data.get("providers")
    if not providers and "agents" in data:
        providers = data["agents"]

    # normalize shape
    if not providers:
        providers = []
    elif isinstance(providers, str):
        providers = [providers]

    providers = [str(p).strip().lower() for p in providers]
    providers = [_ALIAS.get(p, p) for p in providers]

    available = _available_providers_from_env()
    requested = set(providers)
    chosen = (requested & available) if requested else set(available)

    if not chosen and "openai" in available:
        chosen = {"openai"}

    return chosen, available

# =============================================================================
# GPS FOUNDATION RUNTIME CONFIG LOADER
# =============================================================================
RUNTIME_CONFIG = None

def load_runtime_config():
    """Load GPS Foundation runtime configuration (lazy loaded)"""
    global RUNTIME_CONFIG
    if RUNTIME_CONFIG is not None:
        return RUNTIME_CONFIG
        
    config_path = Path("config/gps_runtime_config.yaml")
    if not config_path.exists():
        # Create default config if missing
        config_path.parent.mkdir(exist_ok=True)
        default_config = {
            'runtime': {
                'mock_mode': False,
                'api_timeout': 30,
                'max_agents': 9
            },
            'validation': {
                'require_api_keys': True,
                'preflight_checks': True
            },
            'gps_metadata': {
                'config_version': '1.0.0',
                'coordinate': 'fr_05_uc_11_ec_05_tc_011'
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    with open(config_path) as f:
        RUNTIME_CONFIG = yaml.safe_load(f)
        return RUNTIME_CONFIG

# =============================================================================
# API KEY VALIDATION SYSTEM
# =============================================================================
class APIKeyValidator:
    """GPS Foundation API Key Validation"""
    
    REQUIRED_KEYS = {
        "ANTHROPIC_API_KEY": "Claude/Anthropic",
        "OPENAI_API_KEY": "GPT-4/OpenAI", 
        "GOOGLE_API_KEY": "Gemini/Google",
        "MISTRAL_API_KEY": "Mistral AI",
        "COHERE_API_KEY": "Cohere AI"
    }
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.validation_results = self._validate_all_keys()
    
    def _validate_all_keys(self):
        """Validate all API keys"""
        results = {}
        for key_name, service in self.REQUIRED_KEYS.items():
            key_value = os.getenv(key_name)
            results[key_name] = {
                "service": service,
                "present": bool(key_value and key_value.strip()),
                "masked_value": self._mask_key(key_value) if key_value else None
            }
        return results
    
    def _mask_key(self, key):
        """Mask API key for safe logging"""
        if not key or len(key) < 8:
            return "invalid"
        return f"{key[:4]}...{key[-4:]}"
    
    def get_available_agents(self):
        """Return list of agents with valid API keys"""
        agent_mapping = {
            "ANTHROPIC_API_KEY": "claude",
            "OPENAI_API_KEY": "openai", 
            "GOOGLE_API_KEY": "gemini",
            "MISTRAL_API_KEY": "mistral",
            "COHERE_API_KEY": "cohere"
        }
        
        available = []
        for key_name, agent in agent_mapping.items():
            if self.validation_results[key_name]["present"]:
                available.append(agent)
        return available
    
    def get_status_summary(self):
        """Get summary for status endpoint"""
        available = self.get_available_agents()
        return {
            "total_keys": len(self.REQUIRED_KEYS),
            "available_keys": len(available),
            "available_agents": available,
            "validation_details": {
                key: {
                    "service": info["service"],
                    "status": "✅ Present" if info["present"] else "❌ Missing"
                }
                for key, info in self.validation_results.items()
            }
        }

# =============================================================================
# MOCK RESPONSE CREATOR (ONLY FOR MOCK MODE)
# =============================================================================
def create_mock_response(question, selected_agents):
    """Create mock response when mock_mode is True"""
    mock_responses = []
    
    for i, agent in enumerate(selected_agents):
        mock_responses.append({
            'agent': agent,
            'response': f"[MOCK MODE] {agent.upper()}'s simulated analysis of '{question}': This is a mock response for testing purposes. Real LLM integration would provide actual analysis here.",
            'success': True,
            'tokens': 150 + (i * 50),
            'confidence': 0.8
        })
    
    return {
        'status': 'success',
        'result': {
            'summary_text': f"[MOCK MODE] Expert panel analysis simulation for: {question}. This is not a real analysis.",
            'best_agent': {'agent': selected_agents[0] if selected_agents else 'claude', 'reason': 'Mock selection'},
            'disagreements': [],
            'responses': mock_responses,
            'metadata': {
                'mock_mode': True,
                'gps_coordinate': 'fr_05_uc_11_ec_05_tc_011'
            }
        }
    }

# =============================================================================
# STARTUP INITIALIZATION
# =============================================================================
api_validator = None

def initialize_runtime():
    """Initialize runtime components (called only when running as main)"""
    global api_validator
    
    print("🏛️ Initializing Socrates Expert Panel v4.0...")
    
    # Run bootstrap
    run_bootstrap()

    # Load runtime configuration
    try:
        runtime_config = load_runtime_config()
        print(f"✅ Runtime config loaded - Mock mode: {runtime_config['runtime']['mock_mode']}")
    except Exception as e:
        print(f"❌ Failed to load runtime config: {e}")
        exit(1)

    # Initialize API key validator
    try:
        api_validator = APIKeyValidator()
        available_agents = api_validator.get_available_agents()
        print(f"✅ API Key validation complete - Available agents: {available_agents}")
        
        if not available_agents and not runtime_config['runtime']['mock_mode']:
            print("⚠️ WARNING: No API keys available and mock_mode is False")
            print("   Either set API keys or enable mock_mode for testing")
    except Exception as e:
        print(f"❌ API key validation failed: {e}")
        api_validator = None

def get_api_validator():
    """Get API validator instance, initializing if needed"""
    global api_validator
    if api_validator is None:
        try:
            api_validator = APIKeyValidator()
        except Exception as e:
            print(f"⚠️ Failed to initialize API validator: {e}")
    return api_validator

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/consensus', methods=['POST'])
def expert_panel_analysis():
    try:
        data = request.get_json()
        question = data.get('prompt', '')
        selected_agents = data.get('agents', ['claude', 'openai'])
        
        print(f"🏛️ Received query: {question}")
        print(f"🤖 Selected agents: {selected_agents}")
        
        # ✅ STRICT MOCK MODE CHECK
        runtime_config = load_runtime_config()
        if runtime_config["runtime"]["mock_mode"]:
            print("⚠️ MOCK MODE ACTIVE - Returning simulated responses")
            return jsonify(create_mock_response(question, selected_agents))
        
        # ✅ REAL MODE - Use actual consensus engine
        print("🔥 REAL MODE - Calling actual LLM dispatcher")
        
        # Validate agents have API keys
        validator = get_api_validator()
        if validator:
            available_agents = validator.get_available_agents()
            # Filter selected agents to only include those with API keys
            valid_selected_agents = [agent for agent in selected_agents if agent in available_agents]
            
            if not valid_selected_agents:
                return jsonify({
                    'status': 'error',
                    'error': f'No API keys available for selected agents: {selected_agents}. Available agents: {available_agents}'
                })
            
            selected_agents = valid_selected_agents
            print(f"🔑 Using agents with valid API keys: {selected_agents}")
        
        # Call the REAL consensus engine (no more fake responses!)
        result = generate_expert_panel_response_v3(question, selected_agents)
        
        print(f"🔍 CONSENSUS ENGINE RESULT: {type(result)}")
        print(f"🔍 STATUS: {result.get('status') if isinstance(result, dict) else 'Unknown'}")
        
        # The consensus engine should return the properly formatted result
        # No more fake response injection!
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"❌ Full Error: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/status/keys')
def api_key_status():
    """Status endpoint for API key validation"""
    validator = get_api_validator()
    if validator:
        return jsonify(validator.get_status_summary())
    else:
        return jsonify({"error": "API validator not initialized"})

@app.route('/status/config')
def runtime_config_status():
    """Status endpoint for runtime configuration"""
    runtime_config = load_runtime_config()
    return jsonify({
        "runtime_config": runtime_config,
        "mock_mode": runtime_config["runtime"]["mock_mode"],
        "gps_coordinate": runtime_config.get("gps_metadata", {}).get("coordinate", "unknown")
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/submit_query', methods=['POST'])
def submit_query_alias():
    """
    Enhanced compatibility shim for Socrates v4.1 frontend with provider normalization
    Routes /submit_query -> /api/consensus with format transformation
    GPS Coordinate: fr_05_uc_11_ec_06_tc_004_shim
    """
    # 🎯 START TIMING
    import time
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # PATCH 01B: Normalize providers/agents and enforce sane fallback
        chosen, available = _normalize_providers_from_payload(data)
        if not chosen:
            return jsonify({
                "success": False,
                "error": "no_available_providers",
                "available": sorted(list(available)),
                "responses": [],
                "agents": {}
            }), 400

        # Make sure downstream logic sees canonical key
        data["providers"] = sorted(list(chosen))
        
        # Update the selected_agents to use normalized providers
        selected_agents = data["providers"]
        
        question = data.get('prompt', '')
        
        print(f"🏛️ Received query: {question}")
        print(f"🤖 Normalized providers: {selected_agents}")
        print(f"🔍 Available providers: {sorted(list(available))}")
        
        # ✅ STRICT MOCK MODE CHECK
        runtime_config = load_runtime_config()
        if runtime_config["runtime"]["mock_mode"]:
            print("⚠️ MOCK MODE ACTIVE - Returning simulated responses")
            mock_result = create_mock_response(question, selected_agents)
            # Transform mock result to Socrates format
            return _transform_to_socrates_format(mock_result, start_time)
        
        # ✅ REAL MODE - Use actual consensus engine
        print("🔥 REAL MODE - Calling actual LLM dispatcher")
        
        # Call the REAL consensus engine
        result = generate_expert_panel_response_v3(question, selected_agents)
        
        print(f"🔍 CONSENSUS ENGINE RESULT: {type(result)}")
        print(f"🔍 STATUS: {result.get('status') if isinstance(result, dict) else 'Unknown'}")
        
        # Transform to Socrates format
        return _transform_to_socrates_format(result, start_time)
        
    except Exception as e:
        print(f"❌ Error in submit_query_alias: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Format transformation error: {str(e)}',
            'responses': [],
            'synthesis': 'Server error occurred.',
            'champion': None,
            'metrics': {'response_count': 0, 'champion_score': 0, 'process_time': 0},
            'agents': {}
        }), 500

def _transform_to_socrates_format(original_data, start_time):
    """Transform consensus engine result to Socrates v4.1 format"""
    import time
    
    # Transform to Socrates v4.1 format
    if isinstance(original_data, dict) and original_data.get('status') == 'success':
        
        # Initialize responses and agent scores
        responses = []
        agent_scores = {}
        
        # Extract agent responses from consensus engine result
        if 'result' in original_data and 'responses' in original_data['result']:
            raw_responses = original_data['result']['responses']
            print(f"🔍 Found {len(raw_responses)} raw responses from consensus engine")
            
            for response_data in raw_responses:
                agent_name = response_data.get('agent', 'unknown')
                print(f"🔍 Processing {agent_name}: success={response_data.get('success')}")
                
                # Initialize default score
                score = 50
                
                # Only include successful responses with content
                if response_data.get('success') and response_data.get('response'):
                    content = response_data.get('response', '').strip()
                    
                    if content:
                        word_count = len(content.split())
                        
                        # Calculate score from confidence
                        confidence = response_data.get('confidence', 0.5)
                        if confidence <= 1.0:
                            score = int(confidence * 100)
                        else:
                            score = int(confidence)
                        
                        # Ensure score is in valid range
                        score = max(0, min(100, score))
                        
                        responses.append({
                            'agent': agent_name.replace('_', ' ').title(),
                            'content': content,
                            'word_count': word_count,
                            'score': score,
                            'response_time': response_data.get('response_time', 2.0)
                        })
                        
                        agent_scores[agent_name] = {
                            'score': score,
                            'champion': False
                        }
                        
                        print(f"✅ INCLUDED {agent_name}: {len(content)} chars, score={score}")
                    else:
                        print(f"❌ EXCLUDED {agent_name}: empty content")
                else:
                    print(f"❌ EXCLUDED {agent_name}: success={response_data.get('success')}, has_response={bool(response_data.get('response'))}")
        
        print(f"🔍 FINAL RESPONSES COUNT: {len(responses)}")
        
        # Handle case with no valid responses
        if len(responses) == 0:
            return jsonify({
                'success': False,
                'error': 'No valid agent responses received',
                'responses': [],
                'synthesis': 'All agents failed to provide valid responses.',
                'champion': None,
                'metrics': {'response_count': 0, 'champion_score': 0, 'process_time': 0},
                'agents': {}
            })
        
        # Find champion from available responses
        champion_response = max(responses, key=lambda x: x['score'])
        champion = champion_response['agent']
        
        # Update champion status in agent scores
        for agent_name in agent_scores:
            agent_scores[agent_name]['champion'] = (
                agent_name.replace('_', ' ').title() == champion
            )
        
        # 🎯 CALCULATE ACTUAL PROCESSING TIME
        processing_time = time.time() - start_time
        
        # Build Socrates v4.1 response
        socratic_response = {
            'success': True,
            'responses': responses,
            'synthesis': original_data.get('result', {}).get('summary_text', 
                       f"Expert panel analysis complete with {len(responses)} response(s)."),
            'champion': champion,
            'metrics': {
                'response_count': len(responses),
                'total_words': sum(r['word_count'] for r in responses),
                'total_tokens': original_data.get('result', {}).get('metadata', {}).get('total_tokens', 0),
                'champion_score': champion_response['score'],
                'process_time': round(processing_time, 1)
            },
            'agents': agent_scores
        }
        
        print(f"✅ Transformed to Socrates format: {len(responses)} responses")
        
        return jsonify(socratic_response)
    
    else:
        # Handle error case
        return jsonify({
            'success': False,
            'error': original_data.get('error', 'Unknown error from consensus engine'),
            'responses': [],
            'synthesis': 'Analysis failed.',
            'champion': None,
            'metrics': {'response_count': 0, 'champion_score': 0, 'process_time': 0},
            'agents': {}
        })

# =============================================================================
# PATCH 02: ROUTE ALIAS FOR /api/ask
# =============================================================================
@app.route('/api/ask', methods=['POST'])
def api_ask_alias():
    """Forward to the same logic used by /submit_query"""
    return submit_query_alias()

# =============================================================================
# HEALTH ENDPOINTS (Added for Railway deployment)
# =============================================================================
@app.route('/healthz')
def health_check():
    """Fast health check - always returns 200."""
    return jsonify({"status": "healthy"}), 200

@app.route('/readiness')
def readiness_check():
    """Readiness check - validates environment at request time."""
    try:
        runtime_config = load_runtime_config()
        validator = get_api_validator()
        
        readiness_status = {
            "status": "ready",
            "bootstrap_enabled": os.getenv("BOOTSTRAP_ENABLED", "true").lower() == "true",
            "mock_mode": runtime_config["runtime"]["mock_mode"],
            "config_loaded": True,
            "api_validator": validator is not None
        }
        
        if validator:
            available_agents = validator.get_available_agents()
            readiness_status["available_agents"] = available_agents
            readiness_status["agent_count"] = len(available_agents)
        
        return jsonify(readiness_status), 200
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503
                
# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Initialize runtime components
    initialize_runtime()
    
    print("🚀 Starting Socrates Expert Panel v4.0 server...")
    
    runtime_config = load_runtime_config()
    if runtime_config["runtime"]["mock_mode"]:
        print("⚠️  MOCK MODE ENABLED - Responses will be simulated")
    else:
        print("🔥 REAL MODE ENABLED - Using actual LLM dispatcher")
    
    # ✅ FIXED VERSION FOR RAILWAY:    
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, port=port, host='0.0.0.0')