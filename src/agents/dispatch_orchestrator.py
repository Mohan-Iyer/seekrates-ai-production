# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "LLM Agent Integration"
# script_id: "fr_04_uc_344_ec_01_tc_344"
# gps_coordinate: "fr_04_uc_344_ec_01_tc_344"
# script_name: "dispatch_orchestrator.py"
# template_version: "0002.00.00"
# status: "Production"
# =============================================================================

#!/usr/bin/env python3
# LLM LAW 3 COMPLIANCE: ENFORCED
# Verification: Yang - ChatGPT | Enforcement: 2025-06-29

# =============================================================================
# SCRIPT DNA
# =============================================================================
# template_name: "LLM Dispatch Orchestrator"
# filename: "src/agents/dispatch_orchestrator.py"
# version: "1.0.0"
# author: "Claude"
# creation_date: "2025-06-20"
# last_modified: "2025-06-20"
# project_phase: "Week 6: Consensus Engine Implementation"
# status: "production_ready"
# gps_coordinates: "fr_03_uc_01_ec_06_tc_002"
# dependencies: ["asyncio", "llm_registry.py", "llm wrapper modules"]
# purpose: "Async orchestrator for parallel multi-LLM consensus dispatch"
# notes: "Core coordination layer for decision_referee consensus engine"
# =============================================================================

import asyncio
import importlib
import json
import sqlite3
import uuid
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

# Add project paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.path_resolver import PathResolver

# Import LLM registry
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_clients"))
from llm_registry import LLM_WRAPPERS, DEFAULT_CONFIG, get_default_config, get_all_agent_names

async def dispatch_single_llm(agent_name: str, wrapper_module: Any, prompt: str, 
                            config: Dict[str, Any], correlation_id: str,
                            fr_id: str, uc_id: str, ec_id: str, tc_id: str) -> Dict[str, Any]:
    """
    Dispatch prompt to a single LLM wrapper with timeout and error handling
    
    Args:
        agent_name: Name of the LLM agent
        wrapper_module: Imported wrapper module
        prompt: Text prompt to send
        config: Configuration parameters
        correlation_id: Shared correlation ID for tracking
        fr_id, uc_id, ec_id, tc_id: GPS coordinates
        
    Returns:
        Dict with response data or error information
    """
    start_time = datetime.utcnow()
    
    try:
        # Get timeout for this agent
        timeout = config.get('timeout', 30)
        
        # Prepare parameters for the wrapper
        wrapper_params = {
            'prompt': prompt,
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 512),
            'top_p': config.get('top_p', 1.0),
            'fr_id': fr_id,
            'uc_id': uc_id,
            'ec_id': ec_id,
            'tc_id': tc_id
        }
        
        # Execute with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(wrapper_module.query_llm, **wrapper_params),
            timeout=timeout
        )
        
        # Add orchestrator metadata
        response['correlation_id'] = correlation_id
        response['dispatch_timestamp'] = start_time.isoformat() + "Z"
        response['status'] = 'success'
        
        return response
        
    except asyncio.TimeoutError:
        error_response = {
            "agent_name": agent_name,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat() + "Z",
            "dispatch_timestamp": start_time.isoformat() + "Z",
            "response": f"TIMEOUT: {agent_name} exceeded {config.get('timeout', 30)}s timeout",
            "status": "timeout",
            "raw": {"error": "timeout", "timeout_seconds": config.get('timeout', 30)}
        }
        await log_error_to_db(correlation_id, agent_name, "timeout", str(config.get('timeout', 30)))
        return error_response
        
    except Exception as e:
        error_response = {
            "agent_name": agent_name,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat() + "Z",
            "dispatch_timestamp": start_time.isoformat() + "Z",
            "response": f"ERROR: {str(e)}",
            "status": "error",
            "raw": {"error": str(e), "traceback": traceback.format_exc()}
        }
        await log_error_to_db(correlation_id, agent_name, "error", str(e))
        return error_response

async def log_error_to_db(correlation_id: str, agent_name: str, error_type: str, error_details: str):
    """Log orchestrator errors to database"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(project_root, "data", "databases", "llm_transactions.db")
        
        def _log_sync():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            error_payload = json.dumps({
                "error_type": error_type,
                "error_details": error_details,
                "source": "dispatch_orchestrator"
            })
            
            cursor.execute('''
                INSERT INTO llm_responses
                (correlation_id, agent_name, timestamp, fr_id, uc_id, ec_id, tc_id, response_payload, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (correlation_id, agent_name, datetime.utcnow().isoformat(),
                  "fr_03", "uc_01", "ec_06", "tc_002", error_payload, f"Orchestrator {error_type}"))
            
            conn.commit()
            conn.close()
        
        await asyncio.to_thread(_log_sync)
        
    except Exception as e:
        print(f"❌ WARNING: Failed to log orchestrator error: {e}")

def load_wrapper_modules() -> Dict[str, Any]:
    """Dynamically import all LLM wrapper modules"""
    loaded_modules = {}
    
    for agent_name, module_name in LLM_WRAPPERS.items():
        try:
            # Import the wrapper module
            module_path = f"src.agents.llm_clients.{module_name}"
            wrapper_module = importlib.import_module(module_path)
            loaded_modules[agent_name] = wrapper_module
            print(f"✅ Loaded wrapper: {agent_name}")
            
        except ImportError as e:
            print(f"❌ WARNING: Failed to import {agent_name} wrapper: {e}")
            loaded_modules[agent_name] = None
            
    return loaded_modules

async def dispatch_prompt(prompt: str, config: Dict[str, Any] = {}, 
                         fr_id: str = "fr_03", uc_id: str = "uc_01",
                         ec_id: str = "ec_06", tc_id: str = "tc_002",
                         agents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Dispatch a prompt to all available LLM wrappers in parallel
    
    Args:
        prompt: Text prompt to send to all LLMs
        config: Optional configuration overrides (temperature, max_tokens, etc.)
        fr_id, uc_id, ec_id, tc_id: GPS coordinates for tracking
        agents: Optional list of specific agents to use (default: all)
        
    Returns:
        List of response dictionaries from all LLM agents
    """
    # Generate correlation ID for this dispatch session
    correlation_id = str(uuid.uuid4())[:12]  # Longer for orchestrator tracking
    dispatch_start = datetime.utcnow()
    
    print(f"🚀 Starting multi-LLM dispatch with correlation ID: {correlation_id}")
    print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    # Load all wrapper modules
    wrapper_modules = load_wrapper_modules()
    
    # Determine which agents to use
    target_agents = agents if agents else get_all_agent_names()
    available_agents = [agent for agent in target_agents if wrapper_modules.get(agent)]
    
    if not available_agents:
        print("❌ ERROR: No LLM wrappers available")
        return []
    
    print(f"🎯 Targeting {len(available_agents)} agents: {', '.join(available_agents)}")
    
    # Prepare tasks for parallel execution
    tasks = []
    
    for agent_name in available_agents:
        wrapper_module = wrapper_modules[agent_name]
        if wrapper_module is None:
            continue
            
        # Merge default config with provided overrides
        agent_config = get_default_config(agent_name)
        agent_config.update(config)
        
        # Create async task
        task = dispatch_single_llm(
            agent_name=agent_name,
            wrapper_module=wrapper_module,
            prompt=prompt,
            config=agent_config,
            correlation_id=correlation_id,
            fr_id=fr_id,
            uc_id=uc_id,
            ec_id=ec_id,
            tc_id=tc_id
        )
        tasks.append(task)
    
    # Execute all tasks in parallel with error isolation
    print(f"⚡ Executing {len(tasks)} parallel LLM calls...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle exceptions that escaped the individual wrappers
            agent_name = available_agents[i]
            error_result = {
                "agent_name": agent_name,
                "correlation_id": correlation_id,
                "timestamp": dispatch_start.isoformat() + "Z",
                "dispatch_timestamp": dispatch_start.isoformat() + "Z",
                "response": f"EXCEPTION: {str(result)}",
                "status": "exception",
                "raw": {"error": str(result), "exception_type": type(result).__name__}
            }
            processed_results.append(error_result)
        else:
            processed_results.append(result)
    
    # Log dispatch summary
    dispatch_end = datetime.utcnow()
    dispatch_duration = (dispatch_end - dispatch_start).total_seconds()
    
    success_count = len([r for r in processed_results if r.get('status') == 'success'])
    error_count = len(processed_results) - success_count
    
    print(f"✅ Dispatch complete in {dispatch_duration:.2f}s")
    print(f"📊 Results: {success_count} success, {error_count} errors")
    print(f"🆔 Correlation ID: {correlation_id}")
    
    return processed_results

def dispatch_prompt_sync(prompt: str, config: Dict[str, Any] = {}, 
                        fr_id: str = "fr_03", uc_id: str = "uc_01",
                        ec_id: str = "ec_06", tc_id: str = "tc_002",
                        agents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for dispatch_prompt - useful for non-async contexts
    
    Args:
        Same as dispatch_prompt
        
    Returns:
        List of response dictionaries from all LLM agents
    """
    return asyncio.run(dispatch_prompt(prompt, config, fr_id, uc_id, ec_id, tc_id, agents))

if __name__ == "__main__":
    # Test the orchestrator
    print("🧪 Testing LLM Dispatch Orchestrator")
    
    test_prompt = "What is the capital of France? Give a brief answer."
    test_config = {
        "temperature": 0.3,
        "max_tokens": 100
    }
    
    # Run test dispatch
    results = dispatch_prompt_sync(test_prompt, test_config)
    
    print(f"\n📋 Test Results ({len(results)} responses):")
    for result in results:
        status = result.get('status', 'unknown')
        agent = result.get('agent_name', 'unknown')
        response_preview = result.get('response', '')[:100]
        print(f"  {agent}: {status} - {response_preview}{'...' if len(response_preview) == 100 else ''}")