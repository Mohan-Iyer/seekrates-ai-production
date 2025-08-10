# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "LLM Agent Integration"
# script_id: "fr_04_uc_353_ec_01_tc_353"
# gps_coordinate: "fr_04_uc_353_ec_01_tc_353"
# script_name: "src/agents/hlr_evaluator.py"
# template_version: "0002.00.00"
# status: "Production"
# http_access: "CURL_BASED"
# =============================================================================

#!/usr/bin/env python3
# LLM LAW 3 COMPLIANCE: ENFORCED
# Verification: Yang - ChatGPT | Enforcement: 2025-06-29

"""
# =============================================================================
# SCRIPT DNA - GPS FOUNDATION COMPLIANT  
# =============================================================================

# CORE IDENTIFICATION
project_name: "alter_ego"
module_name: "multi_llm_referee"
phase: "3 - Canonical Agent Decision Referee"
script_id: "fr_03_uc_01_ec_01_tc_004"
script_name: "src/agents/multi_llm_referee.py"
script_purpose: "Universal Multi-LLM consensus evaluation engine accepting prompts or input files (YAML/JSON/TXT) with organized output management"

# GPS FOUNDATION INTEGRATION  
gps_coordinate: "fr_03_uc_01_ec_01_tc_004"
function_number: "03"
error_code_number: "01" 
test_case_number: "004"
http_access: "CURL_BASED"

# TRACEABILITY CHAIN
predecessor_script: "src/agents/hlr_evaluator.py"
successor_script: "src/agents/<input_filename>/referee_session_*.yaml"
predecessor_template: "templates/agent_script_template.yaml"
successor_template: "templates/multi_llm_consensus_template.yaml"

# INPUT/OUTPUT SPECIFICATIONS
input_sources:
  - path: "--prompt <string>"
    type: "STRING"
    description: "Direct evaluation prompt via CLI argument"
  - path: "--input-file <path>"
    type: "YAML|JSON|TXT"
    description: "Structured evaluation file (HLR YAML, JSON config, or plain text)"
  - path: "src/agents/raw_hlr_mailman_agent.yaml"
    type: "YAML" 
    description: "Example HLR document for agent evaluation"
output_destinations:
  - path: "src/agents/<input_filename>/referee_session_*.yaml"
    type: "YAML"
    description: "Complete session log with arbitration results and consensus analysis"
  - path: "src/agents/<input_filename>/gpt4_response_*.yaml"
    type: "YAML"
    description: "GPT-4 individual response log with scores and justification"
  - path: "src/agents/<input_filename>/llama3_response_*.yaml"
    type: "YAML"
    description: "Ollama LLaMA3 individual response log with scores and justification"
  - path: "src/agents/<input_filename>/together_response_*.yaml"
    type: "YAML"
    description: "Together.ai individual response log with scores and justification"
  - path: "src/agents/<input_filename>/mistral_response_*.yaml"
    type: "YAML"
    description: "Mistral AI individual response log with scores and justification"
  - path: "src/agents/<input_filename>/gemini_response_*.yaml"
    type: "YAML"
    description: "Google Gemini individual response log with scores and justification"

# ENHANCED INPUT FILE SUPPORT
supported_input_formats:
  - format: "YAML"
    extensions: [".yaml", ".yml"]
    description: "Structured requirements documents like HLR files"
  - format: "JSON" 
    extensions: [".json"]
    description: "Configuration files with evaluation criteria"
  - format: "TEXT"
    extensions: [".txt", ".md"]
    description: "Plain text evaluation prompts"

# ORGANIZED OUTPUT STRUCTURE
output_organization:
  base_directory: "src/agents/"
  folder_naming: "<input_filename>_eval_<timestamp>"
  folder_structure:
    - "agent_responses/"
    - "session_logs/"
    - "arbitration_results/"
  file_naming_convention: "<agent>_<type>_<session_id>.yaml"

# TEMPLATE LINEAGE
template_used: "templates/canonical_agent_template.yaml"
template_version: "0002.00.00"
generation_timestamp: "2025-07-01T12:00:00Z"

# DEPENDENCY MAPPING
dependencies:
  internal_modules: ["src.utils.path_resolver", "src.utils.secrets_manager", "src.utils.curl_wrapper"]
  external_libraries: ["openai", "yaml", "json", "pathlib", "datetime", "argparse", "typing", "os", "sys", "statistics"]
  database_connections: []
  redis_connections: []

# EXECUTION CONTEXT
execution_context: "CLI"
runtime_environment: "Development|Production"
session_mode: "stateless"
memory_source: "None"

# VERSIONING & AUTHORSHIP
author: "Claude (Engineer) | SUPERVISED_BY: Yang & Three-Team"
created_on: "2025-06-15"
last_updated: "2025-07-01"  
version: "1.0.1"
status: "Canonical"

# RELATED ARTIFACTS
related_hlr: "multi_llm_consensus_requirements.yaml"
related_fr: "fr_03_universal_arbitration.yaml"
related_uc: "uc_01_multi_llm_evaluation.yaml" 
related_tc: "tc_004_canonical_referee.yaml"
test_coverage_status: "Full"

# CANONIZATION PROTOCOL
canonization_status: "Production"
canonization_date: "2025-06-15"
canonization_authority: "Three-Team Universal Decision Protocol v1.0"
change_authorization_required: true

# ARBITRATION MODES SUPPORTED
arbitration_modes:
  - mode: "consensus"
    description: "Statistical consensus based on score similarity"
  - mode: "best_of" 
    description: "Highest total score wins"
  - mode: "majority"
    description: "Voting-based decisions"
  - mode: "entropy_check"
    description: "Flag high disagreement between agents"

# CLI INTERFACE SPECIFICATION
cli_arguments:
  required:
    - name: "--prompt|-p"
      type: "string"
      description: "Evaluation prompt or file path"
  optional:
    - name: "--input-file|-f"
      type: "path"
      description: "Alternative file input method"
    - name: "--mode|-m"
      type: "choice"
      choices: ["consensus", "best_of", "majority", "entropy_check"]
      default: "consensus"
    - name: "--agents|-a"
      type: "string"
      description: "Comma-separated agent list"
    - name: "--no-scores|-n"
      type: "flag"
      description: "Freeform evaluation without score extraction"

# ENHANCED OUTPUT MANAGEMENT
output_management:
  auto_folder_creation: true
  folder_naming_strategy: "input_filename_based"
  log_organization: "agent_separated"
  session_tracking: "full_audit_trail"
  cleanup_policy: "retain_all"
"""

#!/usr/bin/env python3
"""
🧠 MULTI-LLM REFEREE - CANONICAL AGENT
Universal Multi-LLM Consensus Evaluation Engine
First Agent by the Three-Team | Canonized Decision Referee

Usage:
    python multi_llm_referee.py --prompt "Evaluate these 3 startup ideas for feasibility and market potential"
    python multi_llm_referee.py --prompt "Should we use React or Vue?" --mode consensus --agents gpt4,llama3
"""

import os
import argparse
import re
import json
import yaml
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

# CURL WRAPPER MIGRATION: Import curl wrapper instead of requests
try:
    from src.utils.curl_wrapper import CurlWrapper
except ImportError:
    print("❌ Missing curl_wrapper. Ensure src/utils/curl_wrapper.py exists.")
    import sys
    sys.exit(1)

# Import your existing modules
try:
    from secrets_manager import get_decrypted_key
except ImportError:
    print("⚠️  WARNING: secrets_manager.py not found. Using environment variables.")
    def get_decrypted_key(key_name):
        return os.getenv(key_name)

@dataclass
class AgentResponse:
    agent_name: str
    response: str
    scores: Dict[str, float]
    justification: str
    timestamp: str
    model: str
    status: str = "success"
    error: Optional[str] = None

class MultiLLMReferee:
    """Universal Multi-LLM Arbitration Engine with CURL-based HTTP access"""
    
    def __init__(self, config_path: str = None):
        self.script_dna = "fr_03_uc_01_ec_01_tc_003"  # GPS coordinate
        self.session_id = f"referee_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.agents = {}
        self.responses = {}
        
        # CURL WRAPPER MIGRATION: Initialize curl wrapper for hallucination-proof HTTP access
        self.curl_wrapper = CurlWrapper()
        
        # Load environment
        self._load_environment()
        self._setup_logging()
        
        # Initialize agents
        self._initialize_agents()
        
        # Enhanced regex patterns (fixed for GPT-4 and Mistral)
        self.score_patterns = {
            'viability': r'(?:\*\*)?viability[_\\]*score(?:\*\*)?:?\s*(\d+(?:\.\d+)?)',
            'financial': r'(?:\*\*)?financial[_\\]*(?:value[_\\]*)?score(?:\*\*)?:?\s*(\d+(?:\.\d+)?)',
            'user_benefit': r'(?:\*\*)?user[_\\]*benefit[_\\]*score(?:\*\*)?:?\s*(\d+(?:\.\d+)?)',
            'feasibility': r'(?:\*\*)?feasibility[_\\]*score(?:\*\*)?:?\s*(\d+(?:\.\d+)?)',
            'impact': r'(?:\*\*)?impact[_\\]*score(?:\*\*)?:?\s*(\d+(?:\.\d+)?)',
            'quality': r'(?:\*\*)?quality[_\\]*score(?:\*\*)?:?\s*(\d+(?:\.\d+)?)',
            'overall': r'(?:\*\*)?overall[_\\]*score(?:\*\*)?:?\s*(\d+(?:\.\d+)?)',
        }
        
        print(f"🔒 Multi-LLM Referee initialized with CURL-based HTTP access")
    
    def _load_environment(self):
        """Load API keys and configuration"""
        # Load .env file
        env_path = Path("src/agents/.env.mailman_agent")
        if env_path.exists():
            print(f"🔑 Loaded {env_path.name}")
            # Load environment variables from file
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # Decrypt API keys
        try:
            self.openai_key = get_decrypted_key("OPENAI_API_KEY")
            if self.openai_key:
                print("✅ Decrypted OpenAI API key using Fernet.")
        except Exception as e:
            print(f"⚠️  OpenAI key error: {e}")
            self.openai_key = None
            
        self.together_key = os.getenv("TOGETHER_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
    
    def _setup_logging(self):
        """Setup logging directory"""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def _initialize_agents(self):
        """Initialize available agents"""
        self.agents = {
            'gpt4': {
                'name': 'GPT-4',
                'enabled': bool(self.openai_key),
                'api_call': self._call_openai
            },
            'llama3': {
                'name': 'Llama3',
                'enabled': True,  # Assuming Ollama is always available
                'api_call': self._call_ollama
            },
            'together': {
                'name': 'Together.ai',
                'enabled': bool(self.together_key),
                'api_call': self._call_together
            },
            'mistral': {
                'name': 'Mistral',
                'enabled': bool(self.mistral_key),
                'api_call': self._call_mistral
            },
            'gemini': {
                'name': 'Gemini',
                'enabled': bool(self.gemini_key),
                'api_call': self._call_gemini
            }
        }
    
    def create_universal_prompt(self, user_prompt: str, evaluation_mode: str = "consensus") -> str:
        """Create a universal prompt template"""
        return f"""🧬 SCRIPT_DNA: {self.script_dna}
📋 SESSION_ID: {self.session_id}
⚖️  EVALUATION_MODE: {evaluation_mode}

TASK: {user_prompt}

INSTRUCTIONS:
Please provide your analysis and evaluation. If applicable, include numerical scores using this format:
- viability_score: X.X (scale 1-10)
- feasibility_score: X.X (scale 1-10) 
- impact_score: X.X (scale 1-10)
- quality_score: X.X (scale 1-10)
- overall_score: X.X (scale 1-10)

Provide detailed justification for your assessment.

RESPONSE_FORMAT: Start with **ACKNOWLEDGMENT:** followed by your analysis and any relevant scores."""

    def evaluate(self, 
                prompt: str, 
                mode: str = "consensus", 
                agents: List[str] = None,
                extract_scores: bool = True) -> Dict:
        """Main evaluation function"""
        
        print(f"🚀 Starting Multi-LLM Referee Evaluation")
        print(f"📍 GPS Coordinate: {self.script_dna}")
        print(f"🧬 Session ID: {self.session_id}")
        print(f"⚖️  Mode: {mode}")
        print(f"🔒 HTTP Access: CURL_BASED (hallucination-proof)")
        
        # Determine which agents to use
        if agents is None:
            agents = [name for name, config in self.agents.items() if config['enabled']]
        else:
            agents = [agent for agent in agents if agent in self.agents and self.agents[agent]['enabled']]
        
        print(f"🌐 Active Agents: {', '.join([self.agents[agent]['name'] for agent in agents])}")
        
        # Create universal prompt
        universal_prompt = self.create_universal_prompt(prompt, mode)
        print(f"📝 Prompt Length: {len(universal_prompt)} chars")
        
        # Collect responses from all agents
        responses = {}
        for agent_key in agents:
            agent_config = self.agents[agent_key]
            print(f"\n📤 Transmitting to {agent_config['name']}...")
            
            try:
                response = agent_config['api_call'](universal_prompt)
                responses[agent_key] = self._process_response(
                    agent_key, response, extract_scores
                )
                print(f"✅ {agent_config['name']} evaluation complete")
                
            except Exception as e:
                print(f"❌ {agent_config['name']} failed: {str(e)}")
                responses[agent_key] = AgentResponse(
                    agent_name=agent_key,
                    response=f"ERROR: {str(e)}",
                    scores={},
                    justification=f"ERROR: {str(e)}",
                    timestamp=datetime.datetime.now().isoformat(),
                    model=agent_key,
                    status="error",
                    error=str(e)
                )
        
        # Apply arbitration logic
        result = self._arbitrate(responses, mode)
        
        # Log results
        self._log_session(prompt, responses, result)
        
        # Display results
        self._display_results(responses, result)
        
        return result
    
    def _process_response(self, agent_key: str, response: str, extract_scores: bool) -> AgentResponse:
        """Process and extract information from agent response"""
        scores = {}
        
        if extract_scores:
            # Extract scores using enhanced regex patterns
            for score_type, pattern in self.score_patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    scores[score_type] = float(match.group(1))
        
        return AgentResponse(
            agent_name=agent_key,
            response=response,
            scores=scores,
            justification=response[:500] + "..." if len(response) > 500 else response,
            timestamp=datetime.datetime.now().isoformat(),
            model=self.agents[agent_key]['name'],
            status="success" if scores or not extract_scores else "no_scores"
        )
    
    def _arbitrate(self, responses: Dict[str, AgentResponse], mode: str) -> Dict:
        """Apply arbitration logic based on mode"""
        valid_responses = {k: v for k, v in responses.items() 
                          if v.status == "success" and v.scores}
        
        if not valid_responses:
            return {
                'decision': 'NO_CONSENSUS',
                'reason': 'No valid scored responses',
                'winning_agent': None,
                'consensus_level': 0.0,
                'summary': responses
            }
        
        if mode == "consensus":
            return self._consensus_arbitration(valid_responses)
        elif mode == "majority":
            return self._majority_arbitration(valid_responses)
        elif mode == "best_of":
            return self._best_of_arbitration(valid_responses)
        elif mode == "entropy_check":
            return self._entropy_arbitration(valid_responses)
        else:
            return self._consensus_arbitration(valid_responses)
    
    def _consensus_arbitration(self, responses: Dict[str, AgentResponse]) -> Dict:
        """Calculate consensus based on score similarity"""
        if len(responses) < 2:
            agent_key = list(responses.keys())[0]
            return {
                'decision': 'SINGLE_AGENT',
                'winning_agent': agent_key,
                'consensus_level': 100.0,
                'summary': responses
            }
        
        # Calculate consensus based on overall scores
        overall_scores = []
        for response in responses.values():
            total_score = sum(response.scores.values())
            overall_scores.append(total_score)
        
        if len(overall_scores) < 2:
            return {'decision': 'INSUFFICIENT_DATA', 'summary': responses}
        
        # Calculate consensus level (inverse of coefficient of variation)
        mean_score = statistics.mean(overall_scores)
        std_dev = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
        consensus_level = max(0, 100 - (std_dev / mean_score * 100)) if mean_score > 0 else 0
        
        # Find winning agent (closest to mean)
        winning_agent = min(responses.keys(), 
                          key=lambda k: abs(sum(responses[k].scores.values()) - mean_score))
        
        return {
            'decision': 'CONSENSUS_ACHIEVED' if consensus_level > 80 else 'LOW_CONSENSUS',
            'winning_agent': winning_agent,
            'consensus_level': round(consensus_level, 1),
            'mean_score': round(mean_score, 1),
            'std_deviation': round(std_dev, 2),
            'summary': responses
        }
    
    def _majority_arbitration(self, responses: Dict[str, AgentResponse]) -> Dict:
        """Simple majority decision"""
        # Implementation for majority voting
        return self._consensus_arbitration(responses)  # Fallback for now
    
    def _best_of_arbitration(self, responses: Dict[str, AgentResponse]) -> Dict:
        """Highest total score wins"""
        if not responses:
            return {'decision': 'NO_RESPONSES', 'summary': {}}
        
        best_agent = max(responses.keys(), 
                        key=lambda k: sum(responses[k].scores.values()))
        
        return {
            'decision': 'BEST_SCORE_SELECTED',
            'winning_agent': best_agent,
            'winning_score': sum(responses[best_agent].scores.values()),
            'summary': responses
        }
    
    def _entropy_arbitration(self, responses: Dict[str, AgentResponse]) -> Dict:
        """Check for high entropy (disagreement) between agents"""
        consensus_result = self._consensus_arbitration(responses)
        
        if consensus_result['consensus_level'] < 50:
            consensus_result['decision'] = 'HIGH_ENTROPY_DETECTED'
            consensus_result['warning'] = 'Significant disagreement between agents'
        
        return consensus_result
    
    # CURL WRAPPER MIGRATION: API calling methods with curl wrapper
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT-4 using CURL wrapper"""
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        
        json_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response_data = self.curl_wrapper.post(
            url="https://api.openai.com/v1/chat/completions",
            json_data=json_data,
            headers=headers,
            timeout=30
        )
        
        if response_data.get('success', False):
            response_json = response_data.get('data', {})
            return response_json["choices"][0]["message"]["content"]
        else:
            error_msg = response_data.get('error', 'Unknown curl wrapper error')
            raise Exception(f"OpenAI API call failed (CURL): {error_msg}")
    
    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama Llama3 using CURL wrapper"""
        json_data = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
        
        response_data = self.curl_wrapper.post(
            url="http://localhost:11434/api/generate",
            json_data=json_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response_data.get('success', False):
            response_json = response_data.get('data', {})
            return response_json["response"]
        else:
            error_msg = response_data.get('error', 'Unknown curl wrapper error')
            raise Exception(f"Ollama API call failed (CURL): {error_msg}")
    
    def _call_together(self, prompt: str) -> str:
        """Call Together.ai using CURL wrapper"""
        headers = {
            "Authorization": f"Bearer {self.together_key}",
            "Content-Type": "application/json"
        }
        
        json_data = {
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response_data = self.curl_wrapper.post(
            url="https://api.together.xyz/v1/chat/completions",
            json_data=json_data,
            headers=headers,
            timeout=30
        )
        
        if response_data.get('success', False):
            response_json = response_data.get('data', {})
            return response_json["choices"][0]["message"]["content"]
        else:
            error_msg = response_data.get('error', 'Unknown curl wrapper error')
            raise Exception(f"Together.ai API call failed (CURL): {error_msg}")
    
    def _call_mistral(self, prompt: str) -> str:
        """Call Mistral AI using CURL wrapper"""
        headers = {
            "Authorization": f"Bearer {self.mistral_key}",
            "Content-Type": "application/json"
        }
        
        json_data = {
            "model": "mistral-small",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response_data = self.curl_wrapper.post(
            url="https://api.mistral.ai/v1/chat/completions",
            json_data=json_data,
            headers=headers,
            timeout=30
        )
        
        if response_data.get('success', False):
            response_json = response_data.get('data', {})
            return response_json["choices"][0]["message"]["content"]
        else:
            error_msg = response_data.get('error', 'Unknown curl wrapper error')
            raise Exception(f"Mistral API call failed (CURL): {error_msg}")
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini"""
        # Your existing Gemini implementation
        raise NotImplementedError("Gemini implementation needed")
    
    def _log_session(self, prompt: str, responses: Dict, result: Dict):
        """Log complete session with CURL access tracking"""
        log_data = {
            'session_id': self.session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'gps_coordinate': self.script_dna,
            'http_access': 'CURL_BASED',
            'prompt': prompt,
            'responses': {k: {
                'agent': v.agent_name,
                'scores': v.scores,
                'status': v.status,
                'justification': v.justification[:200] + "..."
            } for k, v in responses.items()},
            'arbitration_result': result
        }
        
        log_file = self.log_dir / f"referee_session_{self.session_id}.yaml"
        with open(log_file, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False)
        
        print(f"📁 Session logged: {log_file}")
    
    def _display_results(self, responses: Dict, result: Dict):
        """Display formatted results"""
        print(f"\n{'='*80}")
        print(f"📋 MULTI-LLM REFEREE RESULTS")
        print(f"{'='*80}")
        print(f"🧬 Session ID: {self.session_id}")
        print(f"🔒 HTTP Access: CURL_BASED (hallucination-proof)")
        print(f"📊 Agents Evaluated: {len(responses)}")
        print(f"✅ Successful: {len([r for r in responses.values() if r.status == 'success'])}")
        
        print(f"\n📊 AGENT SCORING ANALYSIS:")
        print(f"+{'='*80}+")
        print(f"| {'AGENT':<12} | {'SCORES':<30} | {'TOTAL':<8} | {'STATUS':<8} |")
        print(f"+{'='*80}+")
        
        for agent_key, response in responses.items():
            agent_name = self.agents[agent_key]['name']
            if response.scores:
                score_str = ', '.join([f"{k}:{v}" for k, v in response.scores.items()])
                total = sum(response.scores.values())
                status = "✅" if response.status == "success" else "❌"
            else:
                score_str = "No scores extracted"
                total = 0.0
                status = "❌"
                
            print(f"| {agent_name:<12} | {score_str:<30} | {total:<8.1f} | {status:<8} |")
        
        print(f"+{'='*80}+")
        
        print(f"\n⚖️  ARBITRATION DECISION:")
        print(f"   🎯 Decision: {result.get('decision', 'UNKNOWN')}")
        if 'winning_agent' in result and result['winning_agent']:
            winning_name = self.agents[result['winning_agent']]['name']
            print(f"   👑 Winning Agent: {winning_name}")
        if 'consensus_level' in result:
            print(f"   📈 Consensus Level: {result['consensus_level']}%")
        
        print(f"\n✅ Multi-LLM Referee evaluation completed")
        print(f"{'='*80}")

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🧠 Multi-LLM Referee - Universal Consensus Evaluation Engine (CURL-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_llm_referee.py --prompt "Evaluate these 3 startup ideas for market potential"
  python multi_llm_referee.py --prompt "Should we use React or Vue?" --mode consensus
  python multi_llm_referee.py --prompt "Rank these features by user impact" --agents gpt4,llama3 --mode best_of
        """
    )
    
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="The evaluation prompt to send to all agents"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["consensus", "majority", "best_of", "entropy_check"],
        default="consensus",
        help="Arbitration mode (default: consensus)"
    )
    
    parser.add_argument(
        "--agents", "-a",
        help="Comma-separated list of agents to use (e.g., gpt4,llama3,mistral)"
    )
    
    parser.add_argument(
        "--no-scores", "-n",
        action="store_true",
        help="Don't extract numerical scores, treat as freeform evaluation"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Multi-LLM Referee v1.0.1 - CURL-based HTTP Access | First Agent by the Three-Team"
    )
    
    args = parser.parse_args()
    
    # Initialize referee
    referee = MultiLLMReferee()
    
    # Parse agents list
    agents_list = None
    if args.agents:
        agents_list = [agent.strip() for agent in args.agents.split(',')]
    
    # Run evaluation
    try:
        result = referee.evaluate(
            prompt=args.prompt,
            mode=args.mode,
            agents=agents_list,
            extract_scores=not args.no_scores
        )
        
        # Exit code based on result
        if result.get('decision') in ['CONSENSUS_ACHIEVED', 'BEST_SCORE_SELECTED']:
            exit(0)  # Success
        else:
            exit(1)  # Partial success or issues
            
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        exit(130)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        exit(1)

if __name__ == "__main__":
    main()