# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "Core System Components"
# script_id: "fr_04_uc_343_ec_01_tc_343"
# gps_coordinate: "fr_04_uc_343_ec_01_tc_343"
# script_name: "confidence_scorer.py"
# template_version: "0002.00.00"
# status: "Production"
# =============================================================================

# =============================================================================
# SCRIPT DNA - AUTONOMOUS BUILD AGENT
# =============================================================================
project_name: "Mailman Agent: Autonomous Build Agent"
module_name: "Consensus Engine - Confidence Scoring"
script_id: "fn_03_uc_02_ec_03_tc_030"
script_purpose: "Calculate probabilistic agreement between Agent1/2/3 with semantic similarity scoring"
input_spec: "Dict[str, AgentResponse], session_id: str"
output_spec: "Dict[str, float] with overall/consensus/individual/similarities scores"
previous_script: "response_parser.py"
next_script: "conflict_resolver.py"
related_uc: ["uc_02"]
related_tc: ["tc_030", "tc_031", "tc_032", "tc_033", "tc_034", "tc_035"]
session_mode: "memoryless"
status: "Draft"
# =============================================================================
"""
Confidence Scorer for Multi-Agent Agreement Calculation
Implements probabilistic agreement between Agent1/2/3 with semantic similarity
GPS Coordinates: fn_03_uc_02_ec_03_tc_030
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import re
from collections import Counter

from .consensus_types import AgentResponse, AgentWeight, SimilarityScore
from .consensus_config import ConsensusConfig
from .event_store import EventStore


class ConfidenceScorer:
    """
    GPS: fn_03_uc_02_ec_03_tc_030
    Calculates probabilistic agreement between agents using multiple scoring methods.
    Integrates with existing event store for scoring audit trail.
    """
    
    def __init__(self, config: ConsensusConfig, event_store: EventStore):
        self.config = config
        self.event_store = event_store
        self.logger = logging.getLogger(__name__)
        
        # Similarity weight configuration
        self.similarity_weights = {
            'structural': 0.3,    # Syntax, length, format similarity
            'semantic': 0.5,      # Content meaning similarity  
            'confidence': 0.2     # Agent confidence alignment
        }
    
    async def calculate_scores(
        self,
        responses: Dict[str, AgentResponse],
        session_id: str
    ) -> Dict[str, float]:
        """
        GPS: fn_03_uc_02_ec_03_tc_031
        Calculate comprehensive confidence scores for agent responses.
        """
        try:
            # Log scoring start to event store
            await self.event_store.append_event(
                'consensus_scoring_started',
                {
                    'session_id': session_id,
                    'agent_count': len(responses),
                    'agents': list(responses.keys())
                }
            )
            
            # Calculate pairwise similarities
            similarities = await self._calculate_pairwise_similarities(responses)
            
            # Calculate individual agent confidence scores
            individual_scores = self._calculate_individual_scores(responses)
            
            # Calculate overall agreement score
            overall_agreement = await self._calculate_overall_agreement(
                similarities, individual_scores
            )
            
            # Calculate consensus confidence
            consensus_confidence = self._calculate_consensus_confidence(
                responses, similarities
            )
            
            scores = {
                'overall': overall_agreement,
                'consensus': consensus_confidence,
                'individual': individual_scores,
                'similarities': similarities,
                'session_id': session_id
            }
            
            # Log scoring completion to event store
            await self.event_store.append_event(
                'consensus_scoring_completed',
                {
                    'session_id': session_id,
                    'overall_score': overall_agreement,
                    'consensus_score': consensus_confidence,
                    'individual_scores': individual_scores
                }
            )
            
            return scores
            
        except Exception as e:
            self.logger.error(f"GPS fn_03_uc_02_ec_03: Score calculation failed for session {session_id}: {e}")
            
            # Log error to event store
            await self.event_store.append_event(
                'consensus_scoring_error',
                {
                    'session_id': session_id,
                    'error': str(e),
                    'error_code': 'ec_03'
                }
            )
            
            return self._create_fallback_scores(responses, session_id)
    
    async def calculate_agreement(
        self,
        responses: Dict[str, AgentResponse],
        agent_weights: Dict[str, AgentWeight]
    ) -> float:
        """
        GPS: fn_03_uc_02_ec_03_tc_032
        Calculate weighted agreement score between agents.
        """
        if len(responses) < 2:
            return 1.0  # Single agent always agrees with itself
        
        # Calculate all pairwise similarities
        similarities = await self._calculate_pairwise_similarities(responses)
        
        # Weight similarities by agent importance
        weighted_similarities = []
        total_weight = 0.0
        
        for (agent1, agent2), similarity in similarities.items():
            if agent1 in agent_weights and agent2 in agent_weights:
                weight1 = agent_weights[agent1].weight
                weight2 = agent_weights[agent2].weight
                combined_weight = (weight1 * weight2) ** 0.5  # Geometric mean
                
                weighted_similarities.append(similarity * combined_weight)
                total_weight += combined_weight
        
        if total_weight == 0:
            return 0.5  # Default agreement if no valid weights
        
        return sum(weighted_similarities) / total_weight
    
    async def _calculate_pairwise_similarities(
        self,
        responses: Dict[str, AgentResponse]
    ) -> Dict[Tuple[str, str], float]:
        """GPS: fn_03_uc_02_ec_03_tc_033 - Calculate similarity scores between all agent pairs."""
        similarities = {}
        agents = list(responses.keys())
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i+1:], i+1):
                similarity = await self._calculate_response_similarity(
                    responses[agent1], responses[agent2]
                )
                similarities[(agent1, agent2)] = similarity
        
        return similarities
    
    async def _calculate_response_similarity(
        self,
        response1: AgentResponse,
        response2: AgentResponse
    ) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_034 - Calculate similarity between two agent responses."""
        # Structural similarity
        structural_sim = self._calculate_structural_similarity(
            response1.content, response2.content
        )
        
        # Semantic similarity
        semantic_sim = await self._calculate_semantic_similarity(
            response1.content, response2.content
        )
        
        # Confidence alignment
        confidence_sim = self._calculate_confidence_similarity(
            response1.confidence, response2.confidence
        )
        
        # Weighted combination
        total_similarity = (
            structural_sim * self.similarity_weights['structural'] +
            semantic_sim * self.similarity_weights['semantic'] +
            confidence_sim * self.similarity_weights['confidence']
        )
        
        return total_similarity
    
    def _calculate_structural_similarity(self, content1: str, content2: str) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_035 - Calculate structural similarity."""
        if not content1 or not content2:
            return 0.0
        
        # Length similarity
        len1, len2 = len(content1), len(content2)
        length_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
        
        # Word count similarity
        words1 = content1.split()
        words2 = content2.split()
        word_count_sim = 1.0 - abs(len(words1) - len(words2)) / max(len(words1), len(words2))
        
        # Format pattern similarity
        pattern_sim = self._calculate_pattern_similarity(content1, content2)
        
        return (length_sim + word_count_sim + pattern_sim) / 3
    
    async def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_036 - Calculate semantic similarity."""
        if not content1 or not content2:
            return 0.0
        
        # Jaccard similarity (word overlap)
        jaccard_sim = self._calculate_jaccard_similarity(content1, content2)
        
        # Key concept overlap
        concept_sim = self._calculate_concept_similarity(content1, content2)
        
        return (jaccard_sim + concept_sim) / 2
    
    def _calculate_confidence_similarity(self, conf1: float, conf2: float) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_037 - Calculate confidence score similarity."""
        return 1.0 - abs(conf1 - conf2)
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_038 - Calculate Jaccard similarity coefficient."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_concept_similarity(self, text1: str, text2: str) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_039 - Calculate concept-based similarity."""
        concepts1 = self._extract_key_concepts(text1)
        concepts2 = self._extract_key_concepts(text2)
        
        if not concepts1 and not concepts2:
            return 1.0
        
        common_concepts = len(concepts1.intersection(concepts2))
        total_concepts = len(concepts1.union(concepts2))
        
        return common_concepts / total_concepts if total_concepts > 0 else 0.0
    
    def _extract_key_concepts(self, text: str) -> set:
        """GPS: fn_03_uc_02_ec_03_tc_040 - Extract key concepts from text."""
        words = re.findall(r'\w+', text.lower())
        
        # Filter stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        meaningful_words = [word for word in words if len(word) > 3 and word not in stopwords]
        
        # Return most frequent concepts
        word_counts = Counter(meaningful_words)
        return set(word for word, count in word_counts.most_common(10))
    
    def _calculate_pattern_similarity(self, text1: str, text2: str) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_041 - Calculate formatting pattern similarity."""
        patterns = [
            r'^\s*[-*]\s',      # Bullet points
            r'^\s*\d+\.\s',     # Numbered lists
            r'[A-Z][^.!?]*[.!?]', # Sentences
            r'[A-Z]{2,}',       # Acronyms
            r'\b\d+\b'          # Numbers
        ]
        
        similarities = []
        for pattern in patterns:
            matches1 = len(re.findall(pattern, text1, re.MULTILINE))
            matches2 = len(re.findall(pattern, text2, re.MULTILINE))
            
            if matches1 == 0 and matches2 == 0:
                similarities.append(1.0)
            else:
                total_matches = matches1 + matches2
                similarity = 1.0 - abs(matches1 - matches2) / total_matches
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_individual_scores(
        self,
        responses: Dict[str, AgentResponse]
    ) -> Dict[str, float]:
        """GPS: fn_03_uc_02_ec_03_tc_042 - Calculate individual confidence scores."""
        scores = {}
        
        for agent_id, response in responses.items():
            base_score = response.confidence
            quality_multiplier = self._assess_response_quality(response)
            scores[agent_id] = min(1.0, base_score * quality_multiplier)
        
        return scores
    
    def _assess_response_quality(self, response: AgentResponse) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_043 - Assess response quality multiplier."""
        quality_factors = []
        
        # Length appropriateness
        content_length = len(response.content)
        if 50 <= content_length <= 2000:
            quality_factors.append(1.0)
        elif content_length < 10:
            quality_factors.append(0.3)
        else:
            quality_factors.append(0.8)
        
        # Reasoning quality
        reasoning_length = len(response.reasoning)
        quality_factors.append(1.0 if reasoning_length > 20 else 0.7)
        
        # Error indicators
        if '[ERROR]' in response.content or '[PARSE_ERROR]' in response.content:
            quality_factors.append(0.2)
        else:
            quality_factors.append(1.0)
        
        return sum(quality_factors) / len(quality_factors)
    
    async def _calculate_overall_agreement(
        self,
        similarities: Dict[Tuple[str, str], float],
        individual_scores: Dict[str, float]
    ) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_044 - Calculate overall agreement score."""
        if not similarities:
            return 0.5
        
        avg_similarity = sum(similarities.values()) / len(similarities)
        avg_confidence = sum(individual_scores.values()) / len(individual_scores)
        
        return (avg_similarity + avg_confidence) / 2
    
    def _calculate_consensus_confidence(
        self,
        responses: Dict[str, AgentResponse],
        similarities: Dict[Tuple[str, str], float]
    ) -> float:
        """GPS: fn_03_uc_02_ec_03_tc_045 - Calculate consensus confidence."""
        if len(responses) < 2:
            return 1.0
        
        min_similarity = min(similarities.values()) if similarities else 0.0
        
        # Adjust based on response variance
        confidences = [r.confidence for r in responses.values()]
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        variance_penalty = min(0.3, confidence_variance)
        
        return max(0.0, min_similarity - variance_penalty)
    
    def _create_fallback_scores(
        self,
        responses: Dict[str, AgentResponse],
        session_id: str
    ) -> Dict[str, float]:
        """GPS: fn_03_uc_02_ec_03_tc_046 - Create fallback scores when calculation fails."""
        individual_scores = {
            agent_id: response.confidence
            for agent_id, response in responses.items()
        }
        
        avg_confidence = sum(individual_scores.values()) / len(individual_scores)
        
        return {
            'overall': avg_confidence,
            'consensus': avg_confidence * 0.8,
            'individual': individual_scores,
            'similarities': {},
            'session_id': session_id,
            'fallback': True
        }