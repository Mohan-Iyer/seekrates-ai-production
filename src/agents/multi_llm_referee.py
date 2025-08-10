#!/usr/bin/env python3

# =============================================================================
# SCRIPT DNA METADATA - Following Script DNA Template v0001.02.00
# =============================================================================

# CORE IDENTIFICATION
project_name: "alter_ego"
module_name: "Multi-LLM Referee - Async Orchestrator"
phase: "3 - Decision Referee Agentization"
script_id: "fr_03_uc_01_ec_01_tc_201"
script_name: "src/agents/multi_llm_referee.py"
script_purpose: "Async multi-agent consensus orchestrator with decoupled consensus engine integration"

# GPS FOUNDATION INTEGRATION
gps_coordinate: "fr_03_uc_01_ec_01_tc_201"
function_number: "fn_03"
error_code_number: "ec_01"
test_case_number: "tc_201"

# TRACEABILITY CHAIN
predecessor_script: "src/agents/gemini_agent.py"
successor_script: "src/agents/consensus_engine.py"
predecessor_template: "templates/agent_orchestrator_template.yaml"
successor_template: "templates/consensus_pipeline_template.yaml"

# EXECUTION CONTEXT
execution_context: "Orchestrator"
runtime_environment: "Development/Testing/Production"
session_mode: "async_concurrent"
memory_source: "DatabaseManager"

# VERSIONING & AUTHORSHIP
author: "Claude Team Implementation Engineers"
created_on: "2025-06-17"
last_updated: "2025-06-17"
version: "2.1.0"  # Updated for consensus engine integration
status: "Production"
async_refactor: True
consensus_engine_integrated: True  # New flag
canonization_status: "In Progress"

# GPS FOUNDATION FEATURES
# - Concurrent agent execution with asyncio.gather()
# - Isolated timeout handling per agent
# - Decoupled arbitration via consensus_engine.py
# - Structured logging to agent_interactions.db and consensus_results.db
# - YAML session logging with full trace
# - GPS coordinate error tracking
# - Zero hardcoded paths - uses directory mapping


# LLM LAW 3 COMPLIANCE
llm_law_3_compliance: "ENFORCED"
verification_protocols: "mandatory"
enforcement_date: "2025-06-29"
supervisor_validated: "Yang - ChatGPT"
enforcement_purpose: "LLM usage compliance verification"

# =============================================================================
# ASYNC MULTI-LLM REFEREE IMPLEMENTATION (CONSENSUS ENGINE INTEGRATED)
# =============================================================================

"""
🧠 ASYNC MULTI-LLM REFEREE - CONSENSUS ENGINE INTEGRATED
Universal Multi-LLM Consensus Evaluation Engine with Decoupled Arbitration

Features:
- Concurrent agent execution using asyncio.gather()
- Isolated timeout and error handling per agent
- Decoupled arbitration via consensus_engine.py
- Structured logging to GPS Foundation databases
- Full session traceability with YAML logs
- GPS coordinate error tracking

GPS Coordinate: fr_03_uc_01_ec_01_tc_201
"""

import os
import argparse
import re
import json
import yaml
import asyncio
import datetime
import hashlib
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# GPS Foundation database availability check
DATABASE_AVAILABLE = False  # Disabled per user requirements

# GPS Foundation compliant imports - using path resolver
try:
    from src.utils.path_resolver import resolve_path
    PATH_RESOLVER_AVAILABLE = True
except ImportError:
    def resolve_path(path: str) -> str:
        return path
    PATH_RESOLVER_AVAILABLE = False

# Add project root to Python path for GPS Foundation compliance
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import async agent query functions
try:
    from src.agents.llm_dispatcher import call_llm_agent
    AGENTS_AVAILABLE = True
    print("✅ LLM Dispatcher imported successfully")
except ImportError as e:
    print(f"⚠️  WARNING: LLM Dispatcher import failed: {e}")
    AGENTS_AVAILABLE = False

# Backward compatibility wrappers for existing function calls
def claude_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("claude", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def gpt4_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("gpt4", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def mistral_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("mistral", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def ollama_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("ollama", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def gemini_query(prompt, metadata=None):
    """Gemini not yet integrated with dispatcher - placeholder"""
    return "⚠️ Gemini agent not yet integrated with centralized dispatcher"
# Import consensus engine from same directory
try:
    # Add current directory to path for local imports
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from consensus_engine import (
        execute_arbitration,
        AgentResponse
    )
    
    CONSENSUS_ENGINE_AVAILABLE = True
    print("✅ Consensus engine imported successfully")
    
except Exception as e:
    print(f"⚠️  WARNING: Consensus engine import failed: {e}")
    CONSENSUS_ENGINE_AVAILABLE = False
    
    # Fallback AgentResponse definition
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
        metadata: Optional[Dict[str, Any]] = None

# Import secrets manager with environment fallback
# Import secrets manager with environment fallback
try:
    from src.utils.secrets_manager import SecretsManager
    def get_decrypted_key(key_name):
        return SecretsManager.get_api_key(key_name.lower().replace('_api_key', ''))
    print("✅ Secrets manager imported")
except ImportError:
    def get_decrypted_key(key_name):
        return os.getenv(key_name)
    print("⚠️  Using environment variables for API keys")

# Import async agent query functions
# Import async agent query functions
try:
    from src.agents.llm_dispatcher import call_llm_agent
    AGENTS_AVAILABLE = True
    print("✅ LLM Dispatcher imported successfully")
except ImportError as e:
    print(f"⚠️  WARNING: LLM Dispatcher import failed: {e}")
    AGENTS_AVAILABLE = False

# Backward compatibility wrappers for existing function calls
def claude_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("claude", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def gpt4_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("gpt4", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def mistral_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("mistral", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def ollama_query(prompt, metadata=None):
    """Wrapper for backward compatibility - routes through dispatcher"""
    if metadata is None:
        metadata = {}
    result = call_llm_agent("ollama", prompt, metadata)
    return result["response"] if result["success"] else f"Error: {result.get('error', 'Unknown error')}"

def gemini_query(prompt, metadata=None):
    """Gemini not yet integrated with dispatcher - placeholder"""
    return "⚠️ Gemini agent not yet integrated with centralized dispatcher"

# Import consensus engine from same directory
try:
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from consensus_engine import execute_arbitration, AgentResponse
    CONSENSUS_ENGINE_AVAILABLE = True
    print("✅ Consensus engine imported successfully")
    
except Exception as e:
    print(f"⚠️  WARNING: Consensus engine import failed: {e}")
    CONSENSUS_ENGINE_AVAILABLE = False
    
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
        metadata: Optional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)

# Import secrets manager using environment fallback
try:
    from src.utils.secrets_manager import SecretsManager
    def get_decrypted_key(key_name):
        return SecretsManager.get_api_key(key_name.lower().replace('_api_key', ''))
    print("✅ Secrets manager imported")
except ImportError:
    def get_decrypted_key(key_name):
        return os.getenv(key_name)
    print("⚠️  Using environment variables for API keys")
logger = logging.getLogger(__name__)

class AsyncMultiLLMReferee:
    """
    Async Multi-LLM Arbitration Engine with Decoupled Consensus Engine
    
    GPS Foundation Features:
    - Concurrent agent execution with isolated error handling
    - Decoupled arbitration via consensus_engine.py
    - Structured database logging for audit trails
    - YAML session logging with full traceability
    - GPS coordinate error tracking and resolution
    """
    
    def __init__(self, config_path: str = None):
        """Initialize async multi-agent orchestrator with consensus engine"""
        self.script_dna = "fr_03_uc_01_ec_01_tc_201"
        self.session_id = f"async_referee_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.agents = {}
        self.db_manager = None
        
        # Initialize GPS Foundation components
        self._initialize_database()
        self._load_environment()
        self._setup_logging()
        self._initialize_agents()
        
        # Enhanced score extraction patterns
        self.score_patterns = {
            'viability': r'(?:\*\*)?viability[_\s]*(?:score)?(?:\*\*)?[:\s]*(\d+(?:\.\d+)?)(?:/10)?',
            'financial': r'(?:\*\*)?financial[_\s]*(?:value[_\s]*)?(?:score)?(?:\*\*)?[:\s]*(\d+(?:\.\d+)?)(?:/10)?',
            'user_benefit': r'(?:\*\*)?user[_\s]*benefit[_\s]*(?:score)?(?:\*\*)?[:\s]*(\d+(?:\.\d+)?)(?:/10)?',
            'feasibility': r'(?:\*\*)?feasibility[_\s]*(?:score)?(?:\*\*)?[:\s]*(\d+(?:\.\d+)?)(?:/10)?',
            'impact': r'(?:\*\*)?impact[_\s]*(?:score)?(?:\*\*)?[:\s]*(\d+(?:\.\d+)?)(?:/10)?',
            'quality': r'(?:\*\*)?quality[_\s]*(?:score)?(?:\*\*)?[:\s]*(\d+(?:\.\d+)?)(?:/10)?',
            'overall': r'(?:\*\*)?overall[_\s]*(?:score)?(?:\*\*)?[:\s]*(\d+(?:\.\d+)?)(?:/10)?',
        }
        
        # Verify consensus engine availability
        if not CONSENSUS_ENGINE_AVAILABLE:
            logger.warning("⚠️  Consensus engine not available - using fallback arbitration")
    
    def _initialize_database(self):
        """Initialize GPS Foundation database manager"""
        if DATABASE_AVAILABLE:
            try:
                self.db_manager = create_database_manager()
                logger.info("✅ DatabaseManager initialized for GPS Foundation logging")
            except Exception as e:
                logger.warning(f"⚠️  DatabaseManager initialization failed: {e}")
                self.db_manager = None
        else:
            self.db_manager = None
    
    def _load_environment(self):
        """Load API keys and configuration with GPS Foundation paths"""
        # Use path resolver for environment file
        if PATH_RESOLVER_AVAILABLE:
            env_path = resolve_path("src/agents/.env.mailman_agent")
        else:
            env_path = Path("src/agents/.env.mailman_agent")
        
        if env_path.exists():
            logger.info(f"🔑 Loaded {env_path.name}")
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # Load API keys using GPS Foundation secrets manager
        try:
            self.openai_key = get_decrypted_key("OPENAI_API_KEY")
            if self.openai_key:
                logger.info("✅ OpenAI API key loaded via GPS Foundation")
        except Exception as e:
            logger.warning(f"⚠️  OpenAI key error: {e}")
            self.openai_key = None
            
        self.together_key = os.getenv("TOGETHER_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY") 
        self.gemini_key = os.getenv("GEMINI_API_KEY")
    
    def _setup_logging(self):
        """Setup logging with GPS Foundation path resolution"""
        if PATH_RESOLVER_AVAILABLE:
            self.log_dir = resolve_path("logs")
        else:
            self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
    
    def _initialize_agents(self):
        """Initialize async agent configuration"""
        self.agents = {
            'gpt4': {
                'name': 'GPT-4',
                'enabled': bool(self.openai_key) and AGENTS_AVAILABLE,
                'query_func': gpt4_query,
                'timeout': 60
            },
            'mistral': {
                'name': 'Mistral',
                'enabled': bool(self.mistral_key) and AGENTS_AVAILABLE,
                'query_func': mistral_query,
                'timeout': 45
            },
            'ollama': {
                'name': 'Ollama-Llama3',
                'enabled': AGENTS_AVAILABLE,
                'query_func': ollama_query,
                'timeout': 120
            },
            'gemini': {
                'name': 'Gemini',
                'enabled': bool(self.gemini_key) and AGENTS_AVAILABLE,
                'query_func': gemini_query,
                'timeout': 60
            }
        }
        
        enabled_agents = [name for name, config in self.agents.items() if config['enabled']]
        logger.info(f"🌐 Initialized agents: {', '.join(enabled_agents)}")

    async def run_agent(self, name: str, query_func, prompt_dict: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        Async agent wrapper with timeout and error isolation
        
        Args:
            name: Agent name for identification
            query_func: Async query function
            prompt_dict: Prompt dictionary with 'prompt' key
            timeout: Timeout in seconds
            
        Returns:
            Structured response dict with GPS Foundation compliance
        """
        start_time = datetime.datetime.now()
        
        try:
            logger.info(f"🚀 Starting {name} agent (timeout: {timeout}s)")
            
            # Execute with timeout isolation
            result = await asyncio.wait_for(query_func(prompt_dict), timeout=timeout)
            
            # Ensure result has required contract fields
            if not all(field in result for field in ['agent_name', 'response', 'score', 'justification', 'error']):
                logger.warning(f"⚠️  {name} agent returned incomplete response format")
                
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"✅ {name} agent completed in {duration:.2f}s")
            
            # Add GPS Foundation metadata
            result['metadata'] = result.get('metadata', {})
            result['metadata'].update({
                'gps_coordinate': self.script_dna,
                'session_id': self.session_id,
                'duration': duration,
                'timeout_used': timeout
            })
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Timeout after {timeout}s"
            logger.error(f"⏱️ {name} agent timeout: {error_msg}")
            
            return {
                "agent_name": name,
                "response": "",
                "score": None,
                "justification": f"Agent timeout: {error_msg}",
                "error": error_msg,
                "metadata": {
                    'gps_coordinate': self.script_dna,
                    'session_id': self.session_id,
                    'duration': timeout,
                    'timeout_used': timeout,
                    'timeout_exceeded': True
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.error(f"❌ {name} agent error: {error_msg}")
            
            return {
                "agent_name": name,
                "response": "",
                "score": None,
                "justification": f"Agent failed: {error_msg}",
                "error": error_msg,
                "metadata": {
                    'gps_coordinate': self.script_dna,
                    'session_id': self.session_id,
                    'duration': duration,
                    'timeout_used': timeout,
                    'exception_type': type(e).__name__
                }
            }

    def create_universal_prompt(self, user_prompt: str, evaluation_mode: str = "consensus") -> str:
        """Create GPS Foundation compliant universal prompt"""
        return f"""🧬 SCRIPT_DNA: {self.script_dna}
📋 SESSION_ID: {self.session_id}
⚖️  EVALUATION_MODE: {evaluation_mode}
🔄 ASYNC_ORCHESTRATOR: Active
🧠 CONSENSUS_ENGINE: Integrated

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

    async def evaluate(self, 
                      prompt: str, 
                      mode: str = "consensus", 
                      agents: List[str] = None,
                      extract_scores: bool = True) -> Dict:
        """
        Main async evaluation function with decoupled consensus engine
        
        Args:
            prompt: Evaluation prompt
            mode: Arbitration mode (consensus, majority, best_of, entropy_check)
            agents: List of agent names to use (None = all enabled)
            extract_scores: Whether to extract numerical scores
            
        Returns:
            Arbitration result with GPS Foundation metadata
        """
        
        logger.info(f"🚀 Starting Async Multi-LLM Referee Evaluation")
        logger.info(f"📍 GPS Coordinate: {self.script_dna}")
        logger.info(f"🧬 Session ID: {self.session_id}")
        logger.info(f"⚖️  Mode: {mode}")
        logger.info(f"🧠 Consensus Engine: {'Available' if CONSENSUS_ENGINE_AVAILABLE else 'Fallback'}")
        
        # Determine which agents to use
        if agents is None:
            agents = [name for name, config in self.agents.items() if config['enabled']]
        else:
            agents = [agent for agent in agents if agent in self.agents and self.agents[agent]['enabled']]
        
        if not agents:
            logger.error("❌ No enabled agents available")
            return {'decision': 'NO_AGENTS_AVAILABLE', 'error': 'No enabled agents'}
        
        logger.info(f"🌐 Active Agents: {', '.join([self.agents[agent]['name'] for agent in agents])}")
        
        # Create universal prompt
        universal_prompt = self.create_universal_prompt(prompt, mode)
        prompt_dict = {"prompt": universal_prompt}
        logger.info(f"📝 Prompt Length: {len(universal_prompt)} chars")
        
        # Execute all agents concurrently using asyncio.gather()
        logger.info(f"🔄 Executing {len(agents)} agents concurrently...")
        
        agent_tasks = [
            self.run_agent(
                agent_name, 
                self.agents[agent_name]['query_func'], 
                prompt_dict, 
                self.agents[agent_name]['timeout']
            )
            for agent_name in agents
        ]
        
        # Execute with concurrent isolation
        try:
            raw_responses = await asyncio.gather(*agent_tasks, return_exceptions=False)
        except Exception as e:
            logger.error(f"❌ Critical error in agent gathering: {e}")
            return {'decision': 'EXECUTION_FAILED', 'error': str(e)}
        
        # Process responses
        responses = {}
        for i, raw_response in enumerate(raw_responses):
            agent_name = agents[i]
            try:
                processed_response = self._process_response(agent_name, raw_response, extract_scores)
                responses[agent_name] = processed_response
                
                # Log to database if available
                await self._log_agent_interaction(agent_name, processed_response)
                
                status_icon = "✅" if processed_response.status == "success" else "❌"
                logger.info(f"{status_icon} {self.agents[agent_name]['name']} evaluation processed")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {agent_name} response: {e}")
                responses[agent_name] = AgentResponse(
                    agent_name=agent_name,
                    response=f"Processing error: {str(e)}",
                    scores={},
                    justification=f"Response processing failed: {str(e)}",
                    timestamp=datetime.datetime.now().isoformat(),
                    model=self.agents[agent_name]['name'],
                    status="processing_error",
                    error=str(e)
                )
        
        # Apply arbitration using consensus engine
        logger.info(f"⚖️  Applying {mode} arbitration via consensus_engine.py")
        
        if CONSENSUS_ENGINE_AVAILABLE:
            try:
                result = execute_arbitration(mode, responses)
                logger.info(f"✅ Consensus engine arbitration completed: {result.get('decision', 'UNKNOWN')}")
            except Exception as e:
                logger.error(f"❌ Consensus engine failed: {e}")
                result = self._fallback_arbitration(responses, mode)
        else:
            logger.warning("⚠️  Using fallback arbitration (consensus engine unavailable)")
            result = self._fallback_arbitration(responses, mode)
        
        # Add session metadata
        result.update({
            'session_id': self.session_id,
            'orchestrator_gps': self.script_dna,
            'consensus_engine_used': CONSENSUS_ENGINE_AVAILABLE,
            'total_agents_executed': len(agents),
            'concurrent_execution': True
        })
        
        # Log to database and YAML
        await self._log_consensus_result(prompt, responses, result)
        await self._log_session_yaml(prompt, responses, result)
        
        # Display results
        self._display_results(responses, result)
        
        return result

    def _process_response(self, agent_key: str, raw_response: Dict[str, Any], extract_scores: bool) -> AgentResponse:
        """Process raw agent response into structured format"""
        
        # Handle the new contract format
        response_text = raw_response.get('response', '')
        existing_score = raw_response.get('score')
        justification = raw_response.get('justification', response_text[:500] + "..." if len(response_text) > 500 else response_text)
        error = raw_response.get('error')
        
        scores = {}
        
        if extract_scores and response_text:
            # Extract scores using enhanced regex patterns
            for score_type, pattern in self.score_patterns.items():
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    scores[score_type] = float(match.group(1))
            
            # Include existing score if present
            if existing_score is not None:
                scores['agent_score'] = existing_score
        
        # Determine status
        if error:
            status = "error"
        elif not extract_scores:
            status = "success"
        elif scores:
            status = "success"
        else:
            status = "no_scores"
        
        return AgentResponse(
            agent_name=agent_key,
            response=response_text,
            scores=scores,
            justification=justification,
            timestamp=datetime.datetime.now().isoformat(),
            model=self.agents[agent_key]['name'],
            status=status,
            error=error,
            metadata=raw_response.get('metadata', {})
        )

    def _fallback_arbitration(self, responses: Dict[str, AgentResponse], mode: str) -> Dict:
        """Fallback arbitration when consensus engine is unavailable"""
        valid_responses = {k: v for k, v in responses.items() 
                        if v.status == "success" and v.scores}
        
        base_result = {
            'gps_coordinate': 'fr_03_uc_01_ec_01_tc_201',
            'session_id': self.session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'mode': mode,
            'arbitration_strategy': f'{mode}_fallback',
            'fallback_reason': 'Consensus engine unavailable',
            'total_agents': len(responses),
            'valid_responses': len(valid_responses)
        }
        
        if not valid_responses:
            return {
                **base_result,
                'decision': 'NO_CONSENSUS_FALLBACK',
                'reason': 'No valid responses available for fallback arbitration',
                'winning_agent': None,
                'summary': responses
            }
        
        # Simple best-of fallback logic
        agent_totals = {}
        for agent_name, response in valid_responses.items():
            try:
                total_score = sum(response.scores.values())
                agent_totals[agent_name] = total_score
            except (TypeError, ValueError):
                continue
        
        if agent_totals:
            best_agent = max(agent_totals.keys(), key=lambda k: agent_totals[k])
            return {
                **base_result,
                'decision': 'FALLBACK_BEST_SCORE',
                'winning_agent': best_agent,
                'winning_score': agent_totals[best_agent],
                'agent_scores': agent_totals,
                'summary': responses
            }
        
        return {
            **base_result,
            'decision': 'FALLBACK_FAILED',
            'reason': 'Unable to determine winner via fallback',
            'winning_agent': None,
            'summary': responses
        }
        

    async def _log_agent_interaction(self, agent_name: str, response: AgentResponse):
        """Log individual agent interaction to GPS Foundation database"""
        if not self.db_manager:
            return
        
        try:
            interaction_data = {
                'session_id': self.session_id,
                'gps_coordinate': self.script_dna,
                'agent_name': agent_name,
                'agent_model': response.model,
                'response_text': response.response,
                'scores': json.dumps(response.scores),
                'status': response.status,
                'error': response.error,
                'timestamp': response.timestamp,
                'metadata': json.dumps(response.metadata or {})
            }
            
            # Log to agent_interactions.db
            await self.db_manager.execute(
                "INSERT INTO agent_interactions (session_id, gps_coordinate, agent_name, agent_model, response_text, scores, status, error, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(interaction_data.values())
            )
            
            logger.debug(f"📊 Logged {agent_name} interaction to database")
            
        except Exception as e:
            logger.error(f"❌ Database logging failed for {agent_name}: {e}")

    async def _log_consensus_result(self, prompt: str, responses: Dict, result: Dict):
        """Log consensus result to GPS Foundation database"""
        if not self.db_manager:
            return
        
        try:
            consensus_data = {
                'session_id': self.session_id,
                'gps_coordinate': self.script_dna,
                'prompt': prompt,
                'decision': result.get('decision'),
                'winning_agent': result.get('winning_agent'),
                'consensus_level': result.get('consensus_level'),
                'total_agents': len(responses),
                'successful_agents': len([r for r in responses.values() if r.status == 'success']),
                'timestamp': datetime.datetime.now().isoformat(),
                'arbitration_metadata': json.dumps({
                    'mode': result.get('mode'),
                    'arbitration_strategy': result.get('arbitration_strategy'),
                    'consensus_engine_used': result.get('consensus_engine_used', False),
                    'mean_score': result.get('mean_score'),
                    'std_deviation': result.get('std_deviation'),
                    'warning': result.get('warning')
                })
            }
            
            # Log to consensus_results.db
            await self.db_manager.execute(
                "INSERT INTO consensus_results (session_id, gps_coordinate, prompt, decision, winning_agent, consensus_level, total_agents, successful_agents, timestamp, arbitration_metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(consensus_data.values())
            )
            
            logger.info(f"📊 Logged consensus result to database")
            
        except Exception as e:
            logger.error(f"❌ Consensus database logging failed: {e}")

    async def _log_session_yaml(self, prompt: str, responses: Dict, result: Dict):
        """Log complete session to YAML with full trace"""
        try:
            # Convert responses to serializable format
            responses_data = {}
            for k, v in responses.items():
                responses_data[k] = {
                    'agent': v.agent_name,
                    'model': v.model,
                    'response': v.response,
                    'scores': v.scores,
                    'status': v.status,
                    'error': v.error,
                    'justification': v.justification[:200] + "..." if len(v.justification) > 200 else v.justification,
                    'timestamp': v.timestamp,
                    'metadata': v.metadata
                }
            
            log_data = {
                'gps_coordinate': self.script_dna,
                'session_id': self.session_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'async_orchestrator': True,
                'consensus_engine_integrated': True,
                'consensus_engine_used': result.get('consensus_engine_used', False),
                'prompt': prompt,
                'agents_executed': list(responses.keys()),
                'responses': responses_data,
                'arbitration_result': result,
                'session_metadata': {
                    'total_agents': len(responses),
                    'successful_agents': len([r for r in responses.values() if r.status == 'success']),
                    'error_agents': len([r for r in responses.values() if r.error]),
                    'concurrent_execution': True,
                    'gps_foundation_compliant': True,
                    'consensus_engine_available': CONSENSUS_ENGINE_AVAILABLE
                }
            }
            
            log_file = self.log_dir / f"referee_{self.session_id}.yaml"
            with open(log_file, 'w') as f:
                yaml.dump(log_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"📁 Session logged to YAML: {log_file}")
            
        except Exception as e:
            logger.error(f"❌ YAML session logging failed: {e}")

    def _display_results(self, responses: Dict, result: Dict):
        """Display formatted results with GPS Foundation metadata"""
        print(f"\n{'='*80}")
        print(f"📋 ASYNC MULTI-LLM REFEREE RESULTS (Consensus Engine Integrated)")
        print(f"{'='*80}")
        print(f"🧬 GPS Coordinate: {self.script_dna}")
        print(f"🚀 Session ID: {self.session_id}")
        print(f"🔄 Async Orchestrator: Active")
        print(f"🧠 Consensus Engine: {'Available' if CONSENSUS_ENGINE_AVAILABLE else 'Fallback'}")
        print(f"📊 Agents Executed: {len(responses)}")
        print(f"✅ Successful: {len([r for r in responses.values() if r.status == 'success'])}")
        print(f"❌ Errors: {len([r for r in responses.values() if r.error])}")
        
        print(f"\n📊 CONCURRENT AGENT EXECUTION ANALYSIS:")
        print(f"+{'='*80}+")
        print(f"| {'AGENT':<12} | {'STATUS':<8} | {'SCORES':<25} | {'TOTAL':<8} | {'TIME':<8} |")
        print(f"+{'='*80}+")
        
        for agent_key, response in responses.items():
            agent_name = self.agents[agent_key]['name']
            
            if response.status == "success":
                status_icon = "✅"
                if response.scores:
                    score_str = ', '.join([f"{k}:{v:.1f}" for k, v in list(response.scores.items())[:3]])
                    total = sum(response.scores.values())
                else:
                    score_str = "No scores"
                    total = 0.0
            else:
                status_icon = "❌"
                score_str = f"Error: {response.error[:15]}..." if response.error else "Failed"
                total = 0.0
            
            # Get duration from metadata
            duration = response.metadata.get('duration', 0) if response.metadata else 0
            duration_str = f"{duration:.1f}s"
            
            print(f"| {agent_name:<12} | {status_icon:<8} | {score_str:<25} | {total:<8.1f} | {duration_str:<8} |")
        
        print(f"+{'='*80}+")
        
        print(f"\n⚖️  ARBITRATION DECISION (via {result.get('arbitration_strategy', 'unknown')} strategy):")
        print(f"   🎯 Decision: {result.get('decision', 'UNKNOWN')}")
        if 'winning_agent' in result and result['winning_agent']:
            winning_name = self.agents[result['winning_agent']]['name']
            print(f"   👑 Winning Agent: {winning_name}")
        if 'consensus_level' in result:
            print(f"   📈 Consensus Level: {result['consensus_level']}%")
        if 'warning' in result:
            print(f"   ⚠️  Warning: {result['warning']}")
        if result.get('fallback_reason'):
            print(f"   🔄 Fallback: {result['fallback_reason']}")
        
        print(f"\n✅ Async Multi-LLM Referee evaluation completed")
        print(f"📊 Database Logging: {'Active' if self.db_manager else 'File-only'}")
        print(f"🧠 Consensus Engine: {'Integrated' if CONSENSUS_ENGINE_AVAILABLE else 'Fallback Used'}")
        print(f"{'='*80}")


# CLI Interface (maintains backward compatibility)
def main():
    """CLI entry point with async execution"""
    parser = argparse.ArgumentParser(
        description="🧠 Async Multi-LLM Referee - Concurrent 4-Agent Consensus Evaluation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_llm_referee.py --prompt "Evaluate these 3 startup ideas for market potential"
  python multi_llm_referee.py --prompt "Should we use React or Vue?" --mode consensus
  python multi_llm_referee.py --prompt "Rank these features by user impact" --agents gpt4,ollama,mistral --mode best_of
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
        help="Comma-separated list of agents to use (e.g., gpt4,ollama,mistral,gemini)"
    )
    
    parser.add_argument(
        "--no-scores", "-n",
        action="store_true",
        help="Don't extract numerical scores, treat as freeform evaluation"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Async Multi-LLM Referee v2.1 - Consensus Engine Integrated by Claude Team"
    )
    
    args = parser.parse_args()
    
    async def run_evaluation():
        """Async evaluation runner"""
        # Initialize referee
        referee = AsyncMultiLLMReferee()
        
        # Parse agents list
        agents_list = None
        if args.agents:
            agents_list = [agent.strip() for agent in args.agents.split(',')]
        
        # Run evaluation
        try:
            result = await referee.evaluate(
                prompt=args.prompt,
                mode=args.mode,
                agents=agents_list,
                extract_scores=not args.no_scores
            )
            
            # Exit code based on result
            if result.get('decision') in ['CONSENSUS_ACHIEVED', 'BEST_SCORE_SELECTED']:
                return 0  # Success
            else:
                return 1  # Partial success or issues
                
        except KeyboardInterrupt:
            print("\n❌ Evaluation interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Critical error: {e}")
            return 1
    
    # Run async evaluation
    try:
        exit_code = asyncio.run(run_evaluation())
        exit(exit_code)
    except Exception as e:
        print(f"❌ Async execution failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()