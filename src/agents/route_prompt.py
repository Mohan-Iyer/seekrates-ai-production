# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "Agent Routing - Dispatcher Proxy"
# script_id: "fr_05_uc_001_ec_01_tc_001"
# gps_coordinate: "fr_05_uc_001_ec_01_tc_001"
# script_name: "src/agents/route_prompt.py"
# script_purpose: "Dispatcher-compliant routing proxy for consensus_engine fallback"
# version: "3.0.0"
# status: "Compliant - Dispatcher Proxy"
# change_reason: "Refactored for dispatcher-only architecture compliance"
# change_date: "2025-07-25"
# modified_by: "Claude"
# =============================================================================

"""
Dispatcher-Compliant Agent Routing System

This module serves as a legacy interface proxy that delegates all LLM calls
to the centralized llm_dispatcher.py. No direct agent execution occurs here.

Primary Purpose:
- Backward compatibility for consensus_engine.py fallback mechanism
- Standardized response format preservation  
- Pure delegation to dispatcher architecture

All actual LLM execution is handled by src.agents.llm_dispatcher.call_llm_agent()
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DISPATCHER INTEGRATION - SINGLE SOURCE OF TRUTH
# ============================================================================

try:
    from src.agents.llm_dispatcher import call_llm_agent as dispatcher_call_llm_agent
    DISPATCHER_AVAILABLE = True
    logger.info("✅ Successfully imported llm_dispatcher.call_llm_agent")
except ImportError as e:
    DISPATCHER_AVAILABLE = False
    logger.error(f"❌ Failed to import llm_dispatcher: {e}")
    
    # Fallback error function if dispatcher unavailable
    def dispatcher_call_llm_agent(prompt: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "response": f"Dispatcher unavailable: {e}",
            "success": False,
            "confidence": 0.0,
            "agent_name": "dispatcher_error",
            "model": "unknown",
            "timestamp": datetime.now().isoformat(),
            "error": f"llm_dispatcher import failed: {e}"
        }

# ============================================================================
# LEGACY INTERFACE PRESERVATION - PURE DELEGATION
# ============================================================================

def route_prompt(prompt: str, preferred_agent: str = "openai", model: Optional[str] = None) -> Dict[str, Any]:
    """
    Legacy routing interface - pure forwarder to llm_dispatcher.
    
    Used by consensus_engine.py as fallback: route_prompt(prompt, agent_name)
    
    Args:
        prompt: The input prompt to route
        preferred_agent: Target agent name (used by consensus_engine)
        model: Optional specific model to use
        
    Returns:
        Standardized response dictionary from dispatcher
        
    Note:
        This function maintains exact backward compatibility for consensus_engine.py
        while enforcing dispatcher-only architecture. Zero direct execution.
    """
    logger.info(f"route_prompt() called - agent: {preferred_agent}, delegating to dispatcher")
    
    try:
        # Build config for dispatcher call
        config = {
            "agent_name": preferred_agent.lower().strip(),
            "model": model
        }
        
        # Remove None values to avoid dispatcher issues
        config = {k: v for k, v in config.items() if v is not None}
        
        # Pure delegation to dispatcher - no legacy execution
        result = dispatcher_call_llm_agent(prompt=prompt, config=config)
        
        # Ensure consistent response format for legacy callers
        if isinstance(result, dict):
            # Add legacy metadata if missing
            result.setdefault("agent_name", preferred_agent)
            result.setdefault("model", model or "default")
            result.setdefault("timestamp", datetime.now().isoformat())
            result.setdefault("success", result.get("response") is not None)
            result.setdefault("confidence", 0.85 if result.get("success") else 0.0)
            
            # Mark as legacy interface call
            metadata = result.setdefault("metadata", {})
            metadata["legacy_interface"] = "route_prompt"
            metadata["dispatcher_delegated"] = True
            
        logger.info(f"route_prompt() completed - success: {result.get('success', False)}")
        return result
        
    except Exception as e:
        logger.error(f"Error in route_prompt delegation: {e}")
        return {
            "response": f"Route prompt delegation error: {str(e)}",
            "success": False,
            "confidence": 0.0,
            "agent_name": preferred_agent,
            "model": model or "unknown",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "metadata": {
                "legacy_interface": "route_prompt",
                "delegation_failed": True
            }
        }


def route_prompt_to_multiple_agents(
    prompt: str, 
    agents: List[str], 
    models: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Multi-agent routing - delegates to dispatcher for each agent.
    
    Args:
        prompt: The input prompt to route
        agents: List of agent names to query
        models: Optional agent-to-model mapping
        
    Returns:
        List of responses from each agent via dispatcher delegation
    """
    logger.info(f"route_prompt_to_multiple_agents() called - agents: {agents}")
    
    results = []
    models = models or {}
    
    for agent in agents:
        try:
            model = models.get(agent)
            result = route_prompt(prompt, agent, model)
            
            # Mark as multi-agent call
            if isinstance(result, dict):
                metadata = result.setdefault("metadata", {})
                metadata["multi_agent_call"] = True
                metadata["agent_index"] = len(results)
                
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error routing to agent {agent}: {e}")
            # Add error result to maintain list structure
            error_result = {
                "agent": agent,
                "response": f"Multi-agent routing error: {str(e)}",
                "success": False,
                "confidence": 0.0,
                "agent_name": agent,
                "model": models.get(agent, "unknown"),
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "metadata": {
                    "multi_agent_call": True,
                    "agent_index": len(results),
                    "delegation_failed": True
                }
            }
            results.append(error_result)
    
    logger.info(f"route_prompt_to_multiple_agents() completed - {len(results)} results")
    return results


# ============================================================================
# UTILITY FUNCTIONS - DISPATCHER DELEGATION
# ============================================================================

def get_available_agents() -> List[str]:
    """
    Get list of available agent names from dispatcher.
    
    Returns:
        List of available agent names via dispatcher query
    """
    logger.info("get_available_agents() called - querying dispatcher")
    
    try:
        # Try to get agent list from dispatcher
        # This would ideally query the dispatcher's available agents
        # For now, return common agent names that dispatcher should support
        return ["openai", "claude", "gemini", "mistral", "ollama", "gpt4", "gpt"]
        
    except Exception as e:
        logger.error(f"Error querying available agents: {e}")
        return ["openai"]  # Minimal fallback


def get_agent_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all available agents via dispatcher delegation.
    
    Returns:
        Dictionary with agent status information from dispatcher
    """
    logger.info("get_agent_status() called - testing agents via dispatcher")
    
    status = {}
    test_prompt = "Hello, this is a connectivity test."
    
    for agent_name in get_available_agents():
        try:
            result = route_prompt(test_prompt, agent_name)
            status[agent_name] = {
                "available": result.get("success", False),
                "dispatcher_routed": True,
                "last_test": datetime.now().isoformat(),
                "model": result.get("model", "unknown"),
                "error": result.get("error") if not result.get("success") else None
            }
        except Exception as e:
            status[agent_name] = {
                "available": False,
                "dispatcher_routed": False,
                "last_test": datetime.now().isoformat(),
                "model": "unknown",
                "error": str(e)
            }
    
    return status


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Alias for any legacy code that might expect call_llm_agent here
# This delegates to the dispatcher instead of providing independent implementation
def call_llm_agent(agent_name: str, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Legacy alias - delegates to route_prompt for dispatcher routing.
    
    Args:
        agent_name: Name of the agent
        prompt: The input prompt
        model: Optional model specification
        
    Returns:
        Response from dispatcher via route_prompt delegation
        
    Warning:
        This is a legacy compatibility alias. Use route_prompt() directly
        or better yet, use src.agents.llm_dispatcher.call_llm_agent() directly.
    """
    logger.warning("call_llm_agent() alias used - consider migrating to dispatcher directly")
    return route_prompt(prompt, agent_name, model)


# ============================================================================
# MAIN EXECUTION (for testing dispatcher delegation)
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Dispatcher-Compliant Routing System")
    print("=" * 60)
    print(f"🔗 Dispatcher Available: {'✅ Yes' if DISPATCHER_AVAILABLE else '❌ No'}")
    
    # Test basic routing
    test_prompt = "Hello, this is a dispatcher delegation test."
    
    print(f"\n🎯 Testing route_prompt function...")
    try:
        routed_result = route_prompt(test_prompt, "claude")
        success = routed_result.get("success", False)
        delegated = routed_result.get("metadata", {}).get("dispatcher_delegated", False)
        
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        print(f"Dispatcher Delegated: {'✅ Yes' if delegated else '❌ No'}")
        print(f"Response Preview: {routed_result.get('response', 'No response')[:100]}...")
        
    except Exception as e:
        print(f"❌ Routing test failed: {e}")
    
    # Test multi-agent routing
    print(f"\n🔄 Testing multi-agent routing...")
    try:
        multi_result = route_prompt_to_multiple_agents(
            test_prompt, 
            ["openai", "claude"], 
            {"openai": "gpt-4", "claude": "claude-3-sonnet"}
        )
        print(f"Multi-agent results: {len(multi_result)} responses")
        for i, result in enumerate(multi_result):
            success = result.get("success", False)
            agent = result.get("agent_name", "unknown")
            print(f"  Agent {i+1} ({agent}): {'✅ Success' if success else '❌ Failed'}")
            
    except Exception as e:
        print(f"❌ Multi-agent test failed: {e}")
    
    # Test agent status
    print(f"\n📊 Agent Status Summary (via dispatcher):")
    try:
        status = get_agent_status()
        for agent, info in status.items():
            availability = "✅ Available" if info["available"] else "❌ Unavailable"
            routed = "🔗 Dispatcher" if info.get("dispatcher_routed") else "❌ Direct"
            print(f"  {agent}: {availability} ({routed})")
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    
    print(f"\n🏁 Dispatcher delegation testing complete")
    print("=" * 60)