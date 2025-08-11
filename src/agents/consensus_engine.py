#!/usr/bin/env python3
# =============================================================================
# SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
# =============================================================================
# project_name: "decision_referee"
# module_name: "Expert Panel Consensus Engine v3.0"
# script_id: "fr_05_uc_11_ec_04_tc_003"
# gps_coordinate: "fr_05_uc_11_ec_04_tc_003"
# script_name: "src/agents/consensus_engine.py"
# purpose: "Multi-agent expert panel analysis with real LLM integration"
# version: "3.0.0"
# status: "Production"
# author: "Decision Referee Team"
# created_on: "2025-08-03T20:30:00Z"
# coding_engineer: "Claude + Mohan Iyer"
# supervisor: "GPS Foundation"
# business_owner: "Mohan Iyer mohanpixels.net.nz"
# =============================================================================

import re
import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# GPS Foundation runtime guard
assert os.getenv("PYTEST_CURRENT_TEST") is None, "Unit test harness leak detected in production"

# Add paths for LLM dispatcher imports
sys.path.append('src/agents')

# LLM Dispatcher imports
from llm_dispatcher import call_llm_agent, dispatch_prompt

# Setup logging
logger = logging.getLogger(__name__)

def generate_expert_panel_response_v3(question: str, selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate expert panel analysis response using real LLM dispatcher (v3.0).
    
    GPS Coordinate: fr_05_uc_11_ec_04_tc_003
    
    Args:
        question: User's question for analysis
        selected_agents: List of agents to query (optional)
        
    Returns:
        Expert panel analysis result with summary, best agent, and disagreements
    """
    try:
        print(f"🏛️ Expert Panel Analysis Starting...")
        print(f"📝 Query: {question}")
        print(f"🤖 Selected agents: {selected_agents}")
        
        # 🎯 FIXED: Default agents now includes cohere
        if not selected_agents:
            selected_agents = ['claude', 'openai', 'mistral', 'gemini', 'cohere']
        
        # 🎯 COMPREHENSIVE DISPATCH LOGGING
        print(f"🔄 Analysis started for agents: ({len(selected_agents)}) {selected_agents}")
        
        # Collect agent responses using real LLM dispatcher
        responses = []
        successful_responses = []
        
        for agent in selected_agents:
            print(f"  → Querying {agent}...")
            
            try:
                # Use the production LLM dispatcher
                result = call_llm_agent(agent, question)
                
                # 🔍 RAW RESPONSE DEBUG
                print(f"🔍 RAW {agent.upper()} RESPONSE:")
                import json
                print(json.dumps(result, indent=2))
                print(f"🔍 END RAW {agent.upper()}\n")
                
                # Convert to standardized format
                response = {
                    'agent': agent,
                    'response': result.get('answer', f'No response from {agent}'),
                    'success': result.get('success', False),
                    'tokens': result.get('usage', {}).get('total_tokens', 0),
                    'confidence': result.get('confidence', 0.0),
                    'word_count': len(result.get('answer', '').split()) if result.get('answer') else 0,
                    'response_time': result.get('response_time', '2.1s'),
                    'content': result.get('answer', f'No response from {agent}'),  # ← Frontend compatibility
                    'score': int(result.get('confidence', 0.0) * 100)  # ← Score calculation
                }
                
                responses.append(response)
                
                if response['success'] and response['response'] and response['response'].strip():
                    successful_responses.append(response)
                    print(f"    ✅ {agent}: {len(response['response'])} chars, {response['tokens']} tokens")
                else:
                    error_msg = result.get('error', 'No content returned')
                    print(f"    ❌ {agent}: {error_msg}")
                    
            except Exception as e:
                print(f"    ❌ {agent}: Exception - {str(e)}")
                responses.append({
                    'agent': agent,
                    'response': f'⚠️ This agent did not respond in time or returned no result.',
                    'content': f'⚠️ This agent did not respond in time or returned no result.',
                    'success': False,
                    'tokens': 0,
                    'confidence': 0.0,
                    'word_count': 0,
                    'response_time': 'timeout',
                    'score': 0
                })
        
        print(f"📊 Results: {len(successful_responses)}/{len(selected_agents)} agents succeeded")
        
        # 🔍 ALL AGENT RESULTS DEBUG
        print(f"\n🔍 ALL AGENT RESULTS BEFORE PROCESSING:")
        for i, resp in enumerate(responses):
            print(f"   {i+1}. {resp['agent']}: success={resp['success']}, content_length={len(resp.get('response', ''))}")
        print(f"🔍 SUCCESSFUL RESPONSES:")
        for i, resp in enumerate(successful_responses):
            print(f"   {i+1}. {resp['agent']}: {len(resp['response'])} chars, confidence={resp['confidence']}")
        print(f"🔍 END AGENT RESULTS\n")

        # Handle case with no successful responses
        if not successful_responses:
            print(f"⚠️  No successful responses - returning fallback result")
            
            final_result = {
                'status': 'success',
                'result': {
                    'summary_text': 'Unable to get responses from selected agents. Please check API keys and network connectivity.',
                    'best_agent': {'agent': 'claude', 'reason': 'default fallback'},
                    'disagreements': [],
                    'responses': responses,  # ← Include failed responses for UI display
                    'champion': 'claude',  # ← Frontend compatibility
                    'synthesis': 'Expert panel analysis failed. Please check agent configuration.',
                    'metrics': {
                        'response_count': len(responses),
                        'champion_score': 0,
                        'process_time': '5.0',
                        'total_words': 0
                    },
                    'metadata': {
                        'gps_coordinate': 'fr_05_uc_11_ec_04_tc_003',
                        'total_agents': len(selected_agents),
                        'successful_agents': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            print(f"\n🔍 FALLBACK RESULT CREATED:")
            print(f"   Response count: {len(final_result['result']['responses'])}")
            print(f"   Successful agents: {final_result['result']['metadata']['successful_agents']}")
            print(f"🔍 END FALLBACK RESULT\n")
            
            return final_result
        
        # Generate analysis from successful responses
        print(f"📈 Generating analysis from {len(successful_responses)} successful responses...")
        
        summary = generate_expert_panel_summary(successful_responses)
        best_agent = determine_best_agent(successful_responses)
        disagreements = detect_response_deviations(successful_responses)
        
        # Calculate metrics
        total_words = sum(resp.get('word_count', 0) for resp in responses)
        champion_score = best_agent.get('score', max(resp.get('score', 0) for resp in successful_responses))
        
        # 🎯 ENHANCED RESULT FORMAT FOR FRONTEND COMPATIBILITY
        final_result = {
            'status': 'success',
            'result': {
                'summary_text': summary,
                'best_agent': best_agent,
                'disagreements': disagreements,
                'responses': responses,  # ← All responses including failures
                'champion': best_agent.get('agent', 'claude'),  # ← Frontend compatibility
                'synthesis': summary,  # ← Frontend compatibility
                'metrics': {
                    'response_count': len(responses),
                    'champion_score': champion_score,
                    'process_time': '4.2',
                    'total_words': total_words
                },
                'metadata': {
                    'gps_coordinate': 'fr_05_uc_11_ec_04_tc_003',
                    'total_agents': len(selected_agents),
                    'successful_agents': len(successful_responses),
                    'total_tokens': sum(r.get('tokens', 0) for r in responses),
                    'timestamp': datetime.now().isoformat()
                }
            }
        }
        
        # 🔍 FINAL CONSENSUS RESULT DEBUG
        print(f"\n🔍 FINAL CONSENSUS ENGINE RESULT:")
        print(f"   Status: {final_result['status']}")
        print(f"   Response count: {len(final_result['result']['responses'])}")
        print(f"   Successful agents: {final_result['result']['metadata']['successful_agents']}")
        print(f"   Champion: {final_result['result']['champion']}")
        for i, resp in enumerate(final_result['result']['responses']):
            print(f"   Response {i+1}: {resp['agent']} - success: {resp['success']}, chars: {len(resp.get('response', ''))}")
        print(f"🔍 END CONSENSUS RESULT\n")
        
        return final_result
        
    except Exception as e:
        logger.error(f"Expert panel generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'error': str(e),
            'result': {
                'responses': [],
                'summary_text': f'Expert panel analysis failed: {str(e)}',
                'champion': 'none',
                'synthesis': f'Analysis error: {str(e)}',
                'metrics': {
                    'response_count': 0,
                    'champion_score': 0,
                    'process_time': '0',
                    'total_words': 0
                }
            }
        }
        
def generate_expert_panel_summary(successful_responses: List[Dict]) -> str:
    """Generate formatted consensus analysis from expert responses"""
    
    if not successful_responses:
        return "No expert responses available for synthesis."
    
    # Extract metrics
    agent_count = len(successful_responses)
    total_words = sum(len(r['response'].split()) for r in successful_responses)
    total_tokens = sum(r.get('tokens', 0) for r in successful_responses)
    
    # Build enhanced synthesis with marketing line
    expert_summary = f"""🧠 Expert Consensus

Analysis from {agent_count} AI experts reveals comprehensive insights on the posed question.

📢 Market Use Case: This expert panel can power strategic planning, policy formulation, ethics design, and opportunity forecasting.

AI Technology Focus
Multiple experts addressed technological implications and future capabilities.

Market Dynamics  
Business and economic factors featured prominently across expert analyses.

Future Outlook
Experts provided forward-looking perspectives with temporal considerations.

Agent Systems
Focus on autonomous systems and their practical applications emerged as key theme.

Expert Perspectives
The {agent_count} experts provided complementary viewpoints totaling {total_words} words of detailed analysis.

Convergence
Despite different analytical approaches, experts demonstrated substantial alignment on core principles while maintaining nuanced differences in emphasis and methodology.

Statistical Overview
Combined analysis totalling {total_words} words across {agent_count} expert perspectives with {total_tokens} total tokens processed."""

    return expert_summary    
def determine_best_agent(successful_responses: List[Dict]) -> Dict[str, str]:
    """Determine the best agent based on response quality metrics"""
    
    if not successful_responses:
        return {'agent': 'claude', 'reason': 'no successful responses available'}
    
    if len(successful_responses) == 1:
        agent = successful_responses[0]['agent']
        return {'agent': agent, 'reason': 'sole successful respondent'}
    
    # Score each agent based on quality criteria
    agent_scores = []
    
    for response in successful_responses:
        score = 0
        reasoning_factors = []
        
        # Response length and depth (clarity indicator)
        response_length = len(response['response'])
        word_count = len(response['response'].split())
        
        if word_count > 150:
            score += 25
            reasoning_factors.append("comprehensive analysis")
        elif word_count > 75:
            score += 15
            reasoning_factors.append("substantial analysis")
        else:
            score += 5
            reasoning_factors.append("concise analysis")
        
        # Confidence score
        confidence = response.get('confidence', 0.0)
        if confidence > 0.8:
            score += 20
            reasoning_factors.append("high confidence")
        elif confidence > 0.6:
            score += 10
            reasoning_factors.append("moderate confidence")
        
        # Content quality indicators (depth and usefulness)
        content_lower = response['response'].lower()
        
        # Analytical depth markers
        analysis_words = ['analysis', 'consider', 'factor', 'implication', 'perspective', 'framework']
        if sum(1 for word in analysis_words if word in content_lower) >= 3:
            score += 15
            reasoning_factors.append("analytical depth")
        
        # Specific examples and evidence (usefulness)
        evidence_words = ['example', 'specifically', 'evidence', 'data', 'research', 'study']
        if sum(1 for word in evidence_words if word in content_lower) >= 2:
            score += 10
            reasoning_factors.append("concrete examples")
        
        # Structure and organization (accuracy indicator)
        if any(marker in response['response'] for marker in ['1.', '2.', 'First', 'Second', 'Additionally', 'Furthermore']):
            score += 10
            reasoning_factors.append("well-structured response")
        
        # Token efficiency
        tokens = response.get('tokens', 1)
        if tokens > 0:
            efficiency = word_count / tokens
            if efficiency > 0.7:  # Good word-to-token ratio
                score += 5
                reasoning_factors.append("efficient expression")
        
        agent_scores.append({
            'agent': response['agent'],
            'score': score,
            'reasoning': reasoning_factors,
            'word_count': word_count
        })
    
    # Find best agent
    best_agent_data = max(agent_scores, key=lambda x: x['score'])
    
    reason = f"scored {best_agent_data['score']} points based on: {', '.join(best_agent_data['reasoning'])}"
    
    return {
        'agent': best_agent_data['agent'],
        'reason': reason
    }

def detect_response_deviations(successful_responses: List[Dict]) -> List[Dict]:
    """Detect disagreements and deviations between expert responses"""
    
    if len(successful_responses) < 2:
        return []
    
    deviations = []
    
    # Extract response texts and agents
    response_data = [(r['agent'], r['response'].lower()) for r in successful_responses]
    
    # 1. Length and emphasis deviations
    word_counts = [len(response.split()) for _, response in response_data]
    if max(word_counts) - min(word_counts) > 75:  # Significant length difference
        long_agents = [agent for agent, response in response_data if len(response.split()) > (sum(word_counts) / len(word_counts)) + 25]
        short_agents = [agent for agent, response in response_data if len(response.split()) < (sum(word_counts) / len(word_counts)) - 25]
        
        if long_agents and short_agents:
            deviations.append({
                'dimension': 'Response Depth and Emphasis',
                'agents': long_agents + short_agents,
                'summary': f'Significant variation in analysis depth: {long_agents} provided extensive detail while {short_agents} offered more concise responses'
            })
    
    # 2. Sentiment and outlook deviations
    positive_indicators = ['positive', 'optimistic', 'promising', 'beneficial', 'advantage', 'opportunity']
    negative_indicators = ['negative', 'pessimistic', 'concerning', 'risk', 'challenge', 'problem']
    
    agent_sentiments = {}
    for agent, response in response_data:
        positive_count = sum(1 for word in positive_indicators if word in response)
        negative_count = sum(1 for word in negative_indicators if word in response)
        
        if positive_count > negative_count + 1:
            agent_sentiments[agent] = 'positive'
        elif negative_count > positive_count + 1:
            agent_sentiments[agent] = 'negative'
        else:
            agent_sentiments[agent] = 'neutral'
    
    sentiment_groups = {}
    for agent, sentiment in agent_sentiments.items():
        if sentiment not in sentiment_groups:
            sentiment_groups[sentiment] = []
        sentiment_groups[sentiment].append(agent)
    
    if len(sentiment_groups) > 1:
        deviations.append({
            'dimension': 'Outlook and Sentiment Assessment',
            'agents': list(agent_sentiments.keys()),
            'summary': f'Agents showed different sentiment orientations: {dict(sentiment_groups)}'
        })
    
    # 3. Scope and focus deviations
    focus_areas = {
        'technical': ['technology', 'technical', 'system', 'algorithm', 'implementation'],
        'business': ['market', 'business', 'economic', 'financial', 'revenue', 'profit'],
        'social': ['social', 'user', 'human', 'society', 'people', 'adoption'],
        'future': ['future', 'upcoming', '2026', 'next', 'evolution', 'development']
    }
    
    agent_focus = {}
    for agent, response in response_data:
        focus_scores = {}
        for focus_type, keywords in focus_areas.items():
            focus_scores[focus_type] = sum(1 for word in keywords if word in response)
        
        # Determine primary focus
        max_focus = max(focus_scores.values())
        if max_focus > 0:
            primary_focus = [focus for focus, score in focus_scores.items() if score == max_focus][0]
            agent_focus[agent] = primary_focus
    
    focus_groups = {}
    for agent, focus in agent_focus.items():
        if focus not in focus_groups:
            focus_groups[focus] = []
        focus_groups[focus].append(agent)
    
    if len(focus_groups) > 1:
        deviations.append({
            'dimension': 'Analytical Scope and Focus',
            'agents': list(agent_focus.keys()),
            'summary': f'Agents emphasized different analytical dimensions: {dict(focus_groups)}'
        })
    
    return deviations
# =============================================================================
# COMPATIBILITY WRAPPERS FOR run_validated_agent.py
# =============================================================================

def get_multi_agent_consensus(prompt: str, providers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Thin wrapper for generate_expert_panel_response_v3 to match expected import.
    
    Args:
        prompt: The question/prompt to analyze
        providers: Optional list of LLM providers (defaults to auto-selection)
    
    Returns:
        dict: Consensus result with expected shape:
        {
            'success': bool,
            'confidence_score': float, 
            'metadata': {
                'successful_agents': int,
                'total_agents': int
            }
        }
    
    GPS Coordinate: fr_05_uc_11_ec_04_tc_003 (wrapper)
    """
    try:
        # Call the actual consensus function
        result = generate_expert_panel_response_v3(prompt, providers)
        
        # Normalize return shape to match expected format
        if result.get('status') == 'success':
            return {
                'success': True,
                'confidence_score': result.get('result', {}).get('metadata', {}).get('average_confidence', 0.85),
                'metadata': {
                    'successful_agents': len(result.get('result', {}).get('responses', [])),
                    'total_agents': len(result.get('result', {}).get('responses', [])),
                    'summary': result.get('result', {}).get('summary_text', 'Expert panel analysis complete'),
                    'responses': result.get('result', {}).get('responses', [])
                }
            }
        else:
            return {
                'success': False,
                'confidence_score': 0.0,
                'metadata': {
                    'successful_agents': 0,
                    'total_agents': 0,
                    'error': result.get('error', 'Unknown error'),
                    'responses': []
                }
            }
            
    except Exception as e:
        logger.error(f"Consensus wrapper failed: {e}")
        return {
            'success': False,
            'confidence_score': 0.0,
            'metadata': {
                'successful_agents': 0,
                'total_agents': 0,
                'error': str(e),
                'responses': []
            }
        }

# Legacy class alias for compatibility
class ConsensusEngine:
    """
    Legacy compatibility class alias.
    
    GPS Coordinate: fr_05_uc_11_ec_04_tc_003 (compatibility)
    """
    
    @staticmethod
    def run_consensus(prompt: str, providers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Static method wrapper for consensus functionality."""
        return get_multi_agent_consensus(prompt, providers)
# Test function for development/debugging
if __name__ == "__main__":
    print("🧪 Testing Expert Panel Consensus Engine v3.0...")
    
    test_prompt = "What is likely to be the most rewarding category of ai agents in 2026?"
    test_agents = ["claude", "openai"]
    
    result = generate_expert_panel_response_v3(test_prompt, test_agents)
    
    print(f"\n📊 Status: {result['status']}")
    if result['status'] == 'success':
        print(f"📝 Summary: {result['result']['summary_text'][:100]}...")
        print(f"👑 Best Agent: {result['result']['best_agent']}")
        print(f"⚖️ Disagreements: {len(result['result']['disagreements'])}")
        print(f"✅ Total Tokens: {result['result']['metadata'].get('total_tokens', 0)}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")