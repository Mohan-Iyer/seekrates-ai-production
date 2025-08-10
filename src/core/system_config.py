
# Enhanced LLM Configuration with Opus 4.1
ENHANCED_LLM_CONFIG = {
    "default_execution_model": "claude-sonnet-4-20250514",
    "complex_reasoning_model": "claude-opus-4-1-20250805", 
    "governance_model": "claude-opus-4-1-20250805",
    
    "routing_rules": {
        "template_governance": "opus",
        "gps_surgical_fixes": "opus", 
        "object_model_analysis": "opus",
        "hallucination_prevention": "opus",
        "multi_agent_coordination": "opus",
        "code_generation": "sonnet",
        "routine_tasks": "sonnet",
        "rapid_execution": "sonnet"
    },
    
    "kubera_integration": {
        "cognitive_framework_model": "opus",  # 40% allocation priority
        "active_operations_model": "sonnet",  # 35% allocation - speed
        "coordination_model": "opus",         # 15% allocation - precision
        "system_overhead_model": "sonnet"     # 10% allocation - efficiency
    }
}

