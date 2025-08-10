# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "LLM Agent Integration"
# script_id: "fr_04_uc_350_ec_01_tc_350"
# gps_coordinate: "fr_04_uc_350_ec_01_tc_350"
# script_name: "base_agent.py"
# template_version: "0002.00.00"
# status: "Production"
# =============================================================================

#!/usr/bin/env python3
"""
# =============================================================================
# GPS FOUNDATION SCRIPT DNA
# =============================================================================
file_type: "python"
file_path: "src/agents/base_agent.py"
filename: "base_agent.py"
project_name: "Mailman Agent: Autonomous Build Agent"
module_name: "Multi-Agent Framework"
phase: "2 - GPS Foundation Implementation"
script_id: "fn_01_uc_02_ec_01_tc_001"
gps_coordinate: "fn_01_uc_02_ec_01_tc_001"
script_purpose: "Abstract base class providing standard interface for all LLM agents"
predecessor_script: "src/core/event_store.py"
successor_script: "src/agents/agent_factory.py"
description: "Foundational abstract base class for all agent implementations"
execution_context: "Agent"
author: "Mohan Iyer (Industrial Engineer + AI Pioneer, 1987)"
created_on: "2025-06-08"
status: "Production"
related_hlr: "hlr_aaa_b_vnext.yaml"
related_fr: "fr_02_multi_agent_framework.yaml"
related_uc: "uc_02 (Multi-Agent Consensus), uc_03 (Agent Validation)"
dependencies: |
  internal_modules: ["src.core.event_store", "src.utils.path_resolver"]
  external_modules: ["abc", "typing", "dataclasses", "datetime", "logging", "asyncio"]
input_spec: "Agent configuration dict, prompt text, context metadata"
output_spec: "AgentResponse object with content, confidence, metadata, timing"
# =============================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import hashlib
import json

# =============================================================================
# GPS FOUNDATION COMPLIANT IMPORTS
# =============================================================================

import sys
import os
from pathlib import Path
import yaml

def get_path(key: str, fallback: str = None) -> str:
    """GPS Foundation compliant path resolver using directory_map.yaml"""
    try:
        # Locate directory_map.yaml from project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        map_file = project_root / "directory_map.yaml"
        
        if map_file.exists():
            with open(map_file, 'r') as f:
                dir_map = yaml.safe_load(f) or {}
            
            # Flatten nested structures
            flattened_map = {}
            for k, v in dir_map.items():
                if isinstance(v, str):
                    flattened_map[k] = v
                elif isinstance(v, dict):
                    for nested_k, nested_v in v.items():
                        if isinstance(nested_v, str):
                            flattened_map[nested_k] = nested_v
            
            return flattened_map.get(key, fallback or key)
        else:
            return fallback or key
            
    except Exception:
        return fallback or key

# GPS Foundation compliant path resolution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add core path using GPS compliant resolution
core_path = Path(get_path("core_modules", "src/core"))
if core_path.exists():
    sys.path.append(str(core_path.parent))

# Import event store using GPS compliant path resolution
try:
    from core.event_store import event_store
except ImportError:
    # Fallback for development environments
    sys.path.append(str(project_root / "src"))
    from core.event_store import event_store

@dataclass
class AgentResponse:
    """
    Standardized response format for all agents in the consensus framework.
    Enables reliable baton passing between agents and consensus validation.
    
    GPS Foundation Integration: Provides structured data for event sourcing
    and multi-agent consensus with surgical precision tracking.
    """
    # Core Response Data
    content: str
    confidence: float  # 0.0-1.0 for consensus weighting
    
    # Agent Metadata
    agent_id: str
    agent_type: str  # 'openai', 'ollama'
    model_name: str
    
    # Performance Metrics
    response_time_ms: float
    token_count: Optional[int] = None
    api_cost: Optional[float] = None
    
    # Consensus Framework Integration
    semantic_fingerprint: str = field(default="")
    validation_scores: Dict[str, float] = field(default_factory=dict)
    
    # Event Sourcing & GPS Foundation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = field(default="")
    request_hash: str = field(default="")
    gps_coordinate: str = field(default="")  # GPS Foundation tracking
    
    # Quality Assurance
    error_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate semantic fingerprint and request hash for consensus tracking."""
        if not self.semantic_fingerprint:
            self.semantic_fingerprint = self._generate_semantic_fingerprint()
        
        if not self.request_hash:
            self.request_hash = hashlib.md5(
                f"{self.content[:100]}{self.timestamp}".encode()
            ).hexdigest()[:8]
    
    def _generate_semantic_fingerprint(self) -> str:
        """
        Generate semantic fingerprint for consensus comparison.
        Used to detect semantic drift between Agent1/2 vs Agent3.
        """
        # Simple semantic fingerprint based on content structure
        content_words = len(self.content.split())
        content_chars = len(self.content)
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        
        return f"{content_words:04d}-{content_chars:05d}-{content_hash}"
    
    def to_event_data(self) -> Dict[str, Any]:
        """Convert response to event store format for GPS Foundation audit trail."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "confidence": self.confidence,
            "response_time_ms": self.response_time_ms,
            "token_count": self.token_count,
            "semantic_fingerprint": self.semantic_fingerprint,
            "validation_scores": self.validation_scores,
            "error_codes": self.error_codes,
            "warnings": self.warnings,
            "content_length": len(self.content),
            "session_id": self.session_id,
            "gps_coordinate": self.gps_coordinate,
            "gps_foundation_compliant": True
        }

@dataclass
class AgentConfig:
    """
    Standardized configuration for all agents.
    Enables template-based agent instantiation via agent_factory.
    
    GPS Foundation Integration: Supports both OpenAI and Ollama agents
    with consistent configuration patterns for industrial-scale deployment.
    """
    # Agent Identity
    agent_id: str
    agent_type: str  # 'openai', 'ollama'
    model_name: str
    
    # Performance Parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Consensus Framework Settings
    confidence_threshold: float = 0.8
    validation_enabled: bool = True
    event_logging: bool = True
    
    # GPS Foundation Settings
    gps_coordinate_tracking: bool = True
    error_surgical_precision: bool = True
    
    # Model-Specific Settings
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Quality Control
    drift_detection: bool = True
    performance_monitoring: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for GPS Foundation event logging."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "confidence_threshold": self.confidence_threshold,
            "validation_enabled": self.validation_enabled,
            "gps_coordinate_tracking": self.gps_coordinate_tracking,
            "error_surgical_precision": self.error_surgical_precision,
            "model_params": self.model_params,
            "gps_foundation_compliant": True
        }

class BaseAgent(ABC):
    """
    Abstract base class for all LLM agents in the multi-agent consensus framework.
    
    GPS Foundation Integration:
    - Standardized interface for OpenAI and Ollama implementations
    - Event sourcing integration for immutable audit trails
    - GPS coordinate error tracking for surgical debugging
    - Industrial-grade quality control and performance monitoring
    
    Provides standardized interface for:
    - Prompt processing with validation
    - Response generation with confidence scoring
    - Event sourcing integration for audit trails
    - Health monitoring and drift detection
    - Error handling with GPS coordinate logging
    
    Industrial Engineering Principle:
    Every agent is a standardized manufacturing component with predictable
    input/output specifications and quality control mechanisms.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize agent with GPS Foundation compliant configuration.
        
        Args:
            config: AgentConfig object with all agent parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"agent.{config.agent_id}")
        self.gps_coordinate = "fn_01_uc_02_ec_01_tc_001"  # Base agent GPS coordinate
        
        # Performance tracking
        self._call_count = 0
        self._total_response_time = 0.0
        self._error_count = 0
        
        # GPS Foundation integration
        self._log_agent_initialization()
    
    def _log_agent_initialization(self):
        """Log agent initialization to event store for GPS Foundation audit trail."""
        if self.config.event_logging:
            try:
                event_store.append_agent_interaction(
                    session_id=f"agent_{self.config.agent_id}_init",
                    agent_id=self.config.agent_id,
                    interaction_type="agent_initialized",
                    request_data={"initialization_request": True},
                    response_data={
                        "agent_config": self.config.to_dict(),
                        "gps_coordinate": self.gps_coordinate,
                        "initialization_time": datetime.now(timezone.utc).isoformat()
                    },
                    metadata={"gps_foundation_compliant": True}
                )
            except Exception as e:
                self.logger.warning(f"Failed to log initialization: {e}")
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Generate response to prompt with confidence scoring.
        
        Args:
            prompt: Input text for the agent to process
            context: Optional context metadata (session_id, previous_responses, etc.)
        
        Returns:
            AgentResponse with content, confidence, and GPS Foundation metadata
            
        Raises:
            AgentError: On unrecoverable agent failures (logged with GPS coordinates)
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return GPS Foundation compliant status information.
        
        Returns:
            Dict with health status, performance metrics, GPS coordinates, and diagnostics
        """
        pass
    
    def validate_prompt(self, prompt: str) -> List[str]:
        """
        Validate prompt before processing with GPS Foundation error codes.
        
        Args:
            prompt: Input text to validate
            
        Returns:
            List of validation error messages with GPS error codes (empty if valid)
        """
        errors = []
        
        if not prompt or not prompt.strip():
            errors.append("gps_ec_01: Empty or whitespace-only prompt")
        
        if len(prompt) > 10000:  # Reasonable limit
            errors.append("gps_ec_02: Prompt exceeds maximum length (10000 chars)")
        
        # Check for potentially problematic content
        if any(char in prompt for char in ['\x00', '\x01', '\x02']):
            errors.append("gps_ec_03: Prompt contains null or control characters")
        
        return errors
    
    # NOTE: Removed unused parameter(s) context [auto-fixed]
    def calculate_confidence(self, response_content: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate confidence score for response (0.0-1.0) for consensus weighting.
        
        Base implementation provides simple heuristics.
        Override in concrete implementations for model-specific confidence.
        
        Args:
            response_content: Generated response text
            context: Optional context for confidence calculation
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Basic heuristics for confidence
        confidence = 0.5  # Base confidence
        
        # Length-based confidence (very short or very long responses are suspicious)
        length = len(response_content.strip())
        if 50 <= length <= 2000:
            confidence += 0.2
        elif length < 10:
            confidence -= 0.3
        
        # Structure-based confidence
        if any(marker in response_content.lower() for marker in ['```', 'step', '1.', '2.', '3.']):
            confidence += 0.1
        
        # Uncertainty markers reduce confidence
        uncertainty_markers = ['maybe', 'perhaps', 'not sure', 'unclear', 'difficult to say']
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response_content.lower())
        confidence -= uncertainty_count * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def log_response(self, response: AgentResponse, prompt: str):
        """
        Log agent response to GPS Foundation event store for consensus tracking and audit.
        
        Args:
            response: AgentResponse object to log
            prompt: Original prompt for context
        """
        if not self.config.event_logging:
            return
        
        try:
            event_data = response.to_event_data()
            event_data.update({
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
                "prompt_length": len(prompt),
                "call_count": self._call_count,
                "base_agent_gps": self.gps_coordinate
            })
            
            event_store.append_agent_interaction(
                session_id=response.session_id or f"agent_{self.config.agent_id}",
                agent_id=self.config.agent_id,
                interaction_type="response_generated",
                request_data={"prompt_hash": event_data["prompt_hash"], "prompt_length": event_data["prompt_length"]},
                response_data=event_data,
                metadata={"gps_foundation_compliant": True}
            )
        except Exception as e:
            self.logger.error(f"Failed to log response: {e}")
            response.warnings.append(f"gps_ec_04: Event logging failed - {str(e)[:50]}")
    
    def log_error(self, error: Exception, prompt: str, context: Dict[str, Any] = None):
        """
        Log error with GPS coordinates for autonomous debugging.
        
        Args:
            error: Exception that occurred
            prompt: Prompt that caused the error
            context: Additional context for debugging
        """
        self._error_count += 1
        
        # Generate GPS coordinates for surgical error tracking
        error_code = f"ec_{self._error_count:02d}"
        function_id = "fn_01"  # base_agent generate_response
        use_case_id = "uc_02"  # multi-agent consensus
        test_case_id = f"tc_{self._call_count:03d}"
        
        gps_coordinates = f"{function_id}_{use_case_id}_{error_code}_{test_case_id}"
        
        error_data = {
            "gps_coordinates": gps_coordinates,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
            "context": context or {},
            "call_count": self._call_count,
            "error_count": self._error_count,
            "base_agent_gps": self.gps_coordinate
        }
        
        try:
            event_store.append_gps_error(
                session_id=context.get("session_id", f"agent_{self.config.agent_id}") if context else f"agent_{self.config.agent_id}",
                gps_coordinate=gps_coordinates,
                error_type=type(error).__name__,
                error_message=str(error),
                fix_strategy={"autonomous_retry": True, "escalate_to_referee": True},
                metadata=error_data
            )
        except Exception as log_error:
            self.logger.critical(f"Failed to log error to GPS Foundation event store: {log_error}")
        
        self.logger.error(f"Agent error [{gps_coordinates}]: {error}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics for GPS Foundation monitoring and consensus weighting.
        
        Returns:
            Dict with performance statistics and GPS Foundation compliance
        """
        avg_response_time = (
            self._total_response_time / self._call_count 
            if self._call_count > 0 else 0.0
        )
        
        error_rate = (
            self._error_count / self._call_count 
            if self._call_count > 0 else 0.0
        )
        
        return {
            "gps_coordinate": self.gps_coordinate,
            "gps_foundation_compliant": True,
            "call_count": self._call_count,
            "avg_response_time_ms": avg_response_time,
            "error_count": self._error_count,
            "error_rate": error_rate,
            "uptime_healthy": error_rate < 0.1,  # <10% error rate considered healthy
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "model_name": self.config.model_name,
            "consensus_ready": error_rate < 0.05  # <5% error rate for consensus participation
        }
    
    def __call__(self, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Callable interface for agent with GPS Foundation error handling and logging.
        
        Args:
            prompt: Input text for processing
            context: Optional context metadata
            
        Returns:
            AgentResponse with generated content and GPS Foundation metadata
        """
        start_time = datetime.now()
        self._call_count += 1
        
        # Set default context
        if context is None:
            context = {}
        
        # Add session tracking
        if "session_id" not in context:
            context["session_id"] = f"session_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Validate prompt
            validation_errors = self.validate_prompt(prompt)
            if validation_errors:
                raise ValueError(f"Prompt validation failed: {'; '.join(validation_errors)}")
            
            # Generate response
            response = self.generate_response(prompt, context)
            
            # Calculate timing
            end_time = datetime.now()
            response.response_time_ms = (end_time - start_time).total_seconds() * 1000
            self._total_response_time += response.response_time_ms
            
            # Set GPS Foundation context
            response.session_id = context["session_id"]
            response.gps_coordinate = self.gps_coordinate
            
            # Log successful response
            self.log_response(response, prompt)
            
            return response
            
        except Exception as e:
            # Log error with GPS coordinates
            self.log_error(e, prompt, context)
            
            # Return error response for graceful degradation
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            error_response = AgentResponse(
                content=f"Agent error: {str(e)[:100]}...",
                confidence=0.0,
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                model_name=self.config.model_name,
                response_time_ms=response_time,
                session_id=context.get("session_id", ""),
                gps_coordinate=self.gps_coordinate,
                error_codes=[f"gps_ec_{self._error_count:02d}"]
            )
            
            return error_response

# GPS Foundation compliant custom exceptions
class AgentError(Exception):
    """Base exception for agent-related errors with GPS coordinate support."""
    
    def __init__(self, message: str, error_code: str = "gps_ec_00", context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

class AgentTimeoutError(AgentError):
    """Agent response timeout error."""
    pass

class AgentValidationError(AgentError):
    """Agent input validation error.""" 
    pass

class AgentConfigurationError(AgentError):
    """Agent configuration error."""
    pass

# GPS Foundation compliant module-level logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("🤖 BaseAgent GPS Foundation interface loaded successfully!")
    print("📋 Ready for OpenAI and Ollama agent implementations")
    print("⚡ GPS-coordinate error tracking enabled")
    print("🏗️ Industrial-grade multi-agent consensus framework initialized")
    print(f"🧬 GPS Coordinate: fn_01_uc_02_ec_01_tc_001")