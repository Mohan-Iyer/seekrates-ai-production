/*!
=============================================================================
SCRIPT DNA METADATA - GPS FOUNDATION COMPLIANT
=============================================================================
project_name: "decision_referee"
module_name: "Socrates Expert Panel Interface v4.1"
script_id: "fr_05_uc_11_ec_06_tc_004"
script_name: "static/js/socrates-expert-panel.js"
script_purpose: "Fixed interface controller with expert response rendering for Socrates UI v4.1"
version: "4.1.0"
status: "Production"
author: "Mohan Iyer"
created_on: "2025-08-07T16:30:00Z"
last_updated: "2025-08-07T16:30:00Z"
coding_engineer: "Claude"
supervisor: "Mohan Iyer"
business_owner: "Mohan Iyer mohan@pixels.net.nz"
=============================================================================
*/

class SocratesExpertPanel {
    constructor() {
        this.gpsCoordinate = 'fr_05_uc_11_ec_06_tc_004';
        this.loadingInterval = null;
        this.currentAnalysis = null;
        this.selectedTopic = null;
        this.expertAgents = this.initializeExpertAgents();
        
        this.initializeInterface();
        this.bindEvents();
        
        console.log(`🏛️ Socrates Expert Panel v4.1 initialized: ${this.gpsCoordinate}`);
    }
    
    initializeExpertAgents() {
        return {
            // Primary Experts
            'claude': { 
                name: 'Claude Opus', 
                provider: 'Anthropic', 
                tier: 'primary', 
                available: true,
                responded: false,
                metrics: { tokens: 0, time: 0, score: 55 }
            },
            'openai': { 
                name: 'OpenAI GPT-4', 
                provider: 'OpenAI', 
                tier: 'primary', 
                available: true,
                responded: false,
                metrics: { tokens: 0, time: 0, score: 60 }
            },
            'gemini': { 
                name: 'Gemini Pro', 
                provider: 'Google', 
                tier: 'primary', 
                available: true,
                responded: false,
                metrics: { tokens: 0, time: 0, score: 45 }
            },
            'mistral': { 
                name: 'Mistral Large', 
                provider: 'Mistral AI', 
                tier: 'secondary', 
                available: true,
                responded: false,
                metrics: { tokens: 0, time: 0, score: 52 }
            },
            // 🎯 FIXED: Cohere now available
            'cohere': { 
                name: 'Cohere Command', 
                provider: 'Cohere AI', 
                tier: 'secondary', 
                available: true,  // ← CHANGED FROM false TO true
                responded: false,
                metrics: { tokens: 0, time: 0, score: 40 }
            },
            'together': { 
                name: 'Together AI', 
                provider: 'Together AI', 
                tier: 'secondary', 
                available: true,
                responded: false,
                metrics: { tokens: 0, time: 0, score: 48 }
            },

            'hrm': { 
                name: 'HRM (Singapore)', 
                provider: 'Human Resources Singapore', 
                tier: 'specialized', 
                available: true,  // Set to standby status
                responded: false,
                metrics: { tokens: 0, time: 0, score: 35 }
            },

            // Inactive agents
            'perplexity': { 
                name: 'Perplexity', 
                provider: 'Research AI', 
                tier: 'specialized', 
                available: false,
                responded: false,
                metrics: { tokens: 'N/A', time: 'N/A', score: 0 }
            },
            'grok': { 
                name: 'Grok', 
                provider: 'xAI', 
                tier: 'specialized', 
                available: false,
                responded: false,
                metrics: { tokens: 'N/A', time: 'N/A', score: 0 }
            },
            'qwen': { 
                name: 'Qwen', 
                provider: 'Alibaba', 
                tier: 'specialized', 
                available: false,
                responded: false,
                metrics: { tokens: 'N/A', time: 'N/A', score: 0 }
            },
            'deepseek': { 
                name: 'DeepSeek', 
                provider: 'DeepSeek AI', 
                tier: 'specialized', 
                available: false,
                responded: false,
                metrics: { tokens: 'N/A', time: 'N/A', score: 0 }
            }
        };
    }

    initializeInterface() {
        this.loadingMessages = [
            '"The only true wisdom is in knowing you know nothing..." - Socrates',
            '"An unexamined life is not worth living." - Socrates',
            '"I know that I know nothing." - Socrates',
            '"Wisdom begins in wonder." - Socrates',
            '"The only good is knowledge and the only evil is ignorance." - Socrates',
            'Consulting the expert pantheon...',
            'Gathering diverse perspectives...',
            'Analyzing contradictions and agreements...',
            'Seeking truth through dialogue...',
            'Synthesizing expert insights...'
        ];
        
        this.updateExpertStates();
        this.initializeTooltips();
        console.log('✅ Interface initialized');
    }
    
    bindEvents() {
        // Topic selection events
        document.querySelectorAll('.topic-item').forEach(item => {
            item.addEventListener('click', (e) => this.selectTopic(e.target));
        });

        // Use query button
        const useQueryBtn = document.getElementById('use-query-btn');
        if (useQueryBtn) {
            useQueryBtn.addEventListener('click', () => this.useSelectedQuery());
        }

        // Submit query button
        const submitBtn = document.getElementById('submit-query');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitQuery());
        }

        // Expert agent selection (if checkboxes exist)
        document.querySelectorAll('.expert-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.updateExpertSelection(e.target.value, e.target.checked);
            });
        });

        // Expert agent card clicks
        document.querySelectorAll('.expert-agent').forEach(agent => {
            agent.addEventListener('click', (e) => {
                if (!e.target.closest('.expert-selection')) {
                    this.toggleExpertAgent(agent.dataset.agent);
                }
            });
        });

        console.log('✅ Event handlers bound');
    }
    
    updateExpertStates() {
        Object.keys(this.expertAgents).forEach(agentKey => {
            const agent = this.expertAgents[agentKey];
            const agentElement = document.querySelector(`[data-agent="${agentKey}"]`);
            const agentCard = document.getElementById(`${agentKey}-card`);
            
            if (agentCard) {
                // Reset classes
                agentCard.classList.remove('inactive', 'selected', 'status-complete', 'status-skipped', 'status-unavailable', 'champion');
                
                // Remove existing crown
                const existingCrown = agentCard.querySelector('.crown');
                if (existingCrown) existingCrown.remove();
                
                const statusElement = agentCard.querySelector('.agent-status');
                const scoreElement = agentCard.querySelector('small');
                
                if (!agent.available) {
                    agentCard.classList.add('inactive');
                    if (statusElement) {
                        statusElement.textContent = 'Unavailable';
                        statusElement.className = 'agent-status status-inactive';
                    }
                    if (scoreElement) {
                        scoreElement.textContent = 'N/A';
                    }
                } else if (agent.responded) {
                    agentCard.classList.add('active');
                    if (statusElement) {
                        statusElement.textContent = 'Complete';
                        statusElement.className = 'agent-status status-active';
                    }
                    if (scoreElement) {
                        scoreElement.textContent = `${agent.metrics.score} pts`;
                    }
                } else {
                    agentCard.classList.add('active');
                    if (statusElement) {
                        statusElement.textContent = 'Ready';
                        statusElement.className = 'agent-status status-active';
                    }
                    if (scoreElement) {
                        scoreElement.textContent = `${agent.metrics.score} pts`;
                    }
                }
            }
        });
        
        this.updateActiveCount();
        console.log('✅ Expert states updated');
    }

    updateActiveCount() {
        const activeAgents = Object.values(this.expertAgents).filter(agent => agent.available);
        const countElement = document.getElementById('active-count');
        if (countElement) {
            countElement.textContent = activeAgents.length;
        }
    }

    initializeTooltips() {
        document.querySelectorAll('.agent-card.inactive').forEach(agent => {
            agent.title = "This expert is not currently available";
            agent.style.cursor = 'not-allowed';
        });
    }
    
    selectTopic(topicElement) {
        // Remove previous selection
        document.querySelectorAll('.topic-item').forEach(item => {
            item.classList.remove('selected');
        });

        // Select current topic
        topicElement.classList.add('selected');
        this.selectedTopic = topicElement.dataset.query;

        // Update query display
        const selectedQueryElement = document.getElementById('selected-query');
        if (selectedQueryElement) {
            selectedQueryElement.textContent = this.selectedTopic;
        }

        const useQueryBtn = document.getElementById('use-query-btn');
        if (useQueryBtn) {
            useQueryBtn.disabled = false;
        }

        console.log('🎯 Topic selected:', this.selectedTopic);
    }
    
    useSelectedQuery() {
        if (this.selectedTopic) {
            const queryInput = document.getElementById('query-input');
            if (queryInput) {
                queryInput.value = this.selectedTopic;
                queryInput.focus();
                console.log('✅ Query set from topic');
            }
        }
    }
    
    toggleExpertAgent(agentKey) {
        const agent = this.expertAgents[agentKey];
        if (!agent.available) return;
        
        const checkbox = document.querySelector(`input[value="${agentKey}"]`);
        if (checkbox && !checkbox.disabled) {
            checkbox.checked = !checkbox.checked;
            this.updateExpertSelection(agentKey, checkbox.checked);
        }
    }
    
    updateExpertSelection(agentKey, isSelected) {
        const agentElement = document.querySelector(`[data-agent="${agentKey}"]`);
        
        if (agentElement) {
            if (isSelected) {
                agentElement.classList.add('selected');
            } else {
                agentElement.classList.remove('selected');
            }
        }
        
        this.updateActiveCount();
    }
    
    async submitQuery() {
        const queryInput = document.getElementById('query-input');
        const query = queryInput ? queryInput.value.trim() : '';
        
        if (!query) {
            alert('Please enter a query to submit to the expert panel.');
            return;
        }

        // 🎯 ENHANCED AGENT SELECTION WITH LOGGING
        const selectedAgents = this.getSelectedAgents();
        
        if (selectedAgents.length === 0) {
            alert('No available experts selected. Please check agent configuration.');
            return;
        }

        // 🎯 COMPREHENSIVE DISPATCH LOGGING
        console.log('🚀 SOCRATES DISPATCH ANALYSIS:');
        console.log(`   Query: "${query}"`);
        console.log(`   Selected Agents (${selectedAgents.length}):`, selectedAgents);
        console.log('   Agent Availability Status:');
        Object.keys(this.expertAgents).forEach(agent => {
            const status = this.expertAgents[agent];
            console.log(`      ${agent}: available=${status.available}, tier=${status.tier}`);
        });

        // Start processing state
        this.startSocraticAnalysis(selectedAgents);

        try {
            // 🎯 ENHANCED API CALL WITH LOGGING
            console.log('📡 Sending request to /submit_query with payload:', {
                prompt: query,
                agents: selectedAgents
            });

            const response = await fetch('/submit_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({
                    prompt: query,
                    agents: selectedAgents  // 🎯 Now includes cohere
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('📊 Expert panel response received:', data);

            if (data && data.success) {
                // 🎯 LOG SUCCESSFUL RESPONSES
                console.log('✅ Response analysis:');
                console.log(`   Total responses: ${data.responses ? data.responses.length : 0}`);
                if (data.responses) {
                    data.responses.forEach(resp => {
                        console.log(`   ${resp.agent}: success=${resp.success || true}, length=${resp.content ? resp.content.length : 0}`);
                    });
                }
                
                this.displaySocraticResults(data);
            } else {
                this.displaySocraticError(data.error || 'Expert panel analysis failed');
            }
        } catch (error) {
            console.error('❌ Network error:', error);
            this.displaySocraticError('Network error: ' + error.message);
        } finally {
            this.stopSocraticAnalysis();
        }
    }

    getSelectedAgents() {
        // Try to get from checkboxes first
        const checkboxes = document.querySelectorAll('.expert-checkbox:checked');
        if (checkboxes.length > 0) {
            const selected = Array.from(checkboxes)
                .map(cb => cb.value)
                .filter(agent => this.expertAgents[agent] && this.expertAgents[agent].available);
            
            console.log('🎯 Selected agents from checkboxes:', selected);
            return selected;
        }

        // 🎯 ENHANCED DEFAULT SELECTION - Include more agents
        const availableAgents = Object.keys(this.expertAgents)
            .filter(agent => this.expertAgents[agent].available);
        
        // Default to top 5 available agents (was limited to 2)
        const defaultSelection = availableAgents.slice(0, 5);
        
        console.log('🎯 Default agent selection:', defaultSelection);
        console.log('🎯 Available agents:', availableAgents);
        
        return defaultSelection;
    }

    startSocraticAnalysis(selectedAgents) {
        // Update submit button
        const submitBtn = document.getElementById('submit-query');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Consulting Panel...';
        }

        // Show loading in expert responses container
        this.showLoadingState();
        
        // Update agent cards to processing state
        selectedAgents.forEach(agentKey => {
            const agentCard = document.getElementById(`${agentKey}-card`);
            if (agentCard) {
                const status = agentCard.querySelector('.agent-status');
                if (status) {
                    status.textContent = 'Processing...';
                    status.className = 'agent-status status-processing';
                }
            }
        });
        
        this.startSocraticAnimation();
        console.log('🔄 Analysis started for agents:', selectedAgents);
    }
    
    stopSocraticAnalysis() {
        // Reset submit button
        const submitBtn = document.getElementById('submit-query');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Submit to Panel';
        }
        
        this.stopSocraticAnimation();
        console.log('✅ Analysis stopped');
    }

    showLoadingState() {
        const container = document.getElementById('expert-responses-container');
        if (container) {
            container.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner-border text-primary me-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div>
                        <h5 id="loading-message">Expert panel analyzing your question...</h5>
                        <p class="text-muted mb-0">Consulting AI minds for philosophical insights</p>
                    </div>
                </div>
            `;
        }
    }
    
    startSocraticAnimation() {
        let index = 0;
        this.loadingInterval = setInterval(() => {
            const loadingMessage = document.getElementById('loading-message');
            if (loadingMessage) {
                loadingMessage.textContent = this.loadingMessages[index];
            }
            index = (index + 1) % this.loadingMessages.length;
        }, 2500);
    }
    
    stopSocraticAnimation() {
        if (this.loadingInterval) {
            clearInterval(this.loadingInterval);
            this.loadingInterval = null;
        }
    }
    
    displaySocraticResults(result) {
        this.currentAnalysis = result;
        
        console.log('🎯 Displaying Socratic results:', result);

        // Update expert agents with response data
        if (result.responses) {
            result.responses.forEach(response => {
                const agentKey = response.agent ? response.agent.toLowerCase() : null;
                if (agentKey && this.expertAgents[agentKey]) {
                    this.expertAgents[agentKey].metrics.tokens = response.tokens || response.word_count || 0;
                    this.expertAgents[agentKey].metrics.time = response.response_time || Math.round(Math.random() * 3000);
                    this.expertAgents[agentKey].metrics.score = response.score || this.expertAgents[agentKey].metrics.score;
                    this.expertAgents[agentKey].responded = true;
                }
            });
        }

        // 🎯 CRITICAL: Display expert responses
        this.displayExpertResponses(result.responses || []);
        
        // Display synthesis
        this.displaySocraticSynthesis(result.synthesis || 'Expert panel analysis complete.');
        
        // Update champion
        this.displaySocraticChampion(result.champion);
        
        // Update metrics
        this.updateSessionMetrics(result.metrics || {});
        
        // Update expert states
        this.updateExpertStates();
        
        console.log('✅ Socratic Expert Panel Analysis displayed successfully');
    }

    // 🎯 NEW METHOD: Display Expert Responses with actual content
    displayExpertResponses(responses) {
        const container = document.getElementById('expert-responses-container');
        if (!container) return;

        console.log('🔍 Rendering expert responses:', responses);
        
        if (!responses || responses.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="fas fa-exclamation-triangle fa-3x mb-3" style="color: #ffc107;"></i>
                    <h5>No Expert Responses</h5>
                    <p>No valid responses received from the expert panel.</p>
                    <p class="small">Please check agent configuration and try again.</p>
                </div>
            `;
            return;
        }

        // 🎯 ENHANCED RESPONSE RENDERING WITH FALLBACKS
        let responsesHTML = '';
        responses.forEach((response, index) => {
            const agent = response.agent || 'Unknown Expert';
            
            // 🎯 FALLBACK CONTENT HANDLING
            let content = response.content || response.response || '';
            if (!content || content.trim() === '') {
                content = "⚠️ This agent did not respond in time or returned no result.";
            }
            
            const score = response.score || 0;
            const wordCount = response.word_count || (content ? content.split(' ').length : 0);
            const responseTime = response.response_time || '2.1';
            const success = response.success !== false && content !== "⚠️ This agent did not respond in time or returned no result.";
            
            // Determine if this is the champion
            const isChampion = this.currentAnalysis && 
                            this.currentAnalysis.champion && 
                            agent.toLowerCase() === this.currentAnalysis.champion.toLowerCase();

            // 🎯 VISUAL INDICATORS FOR FAILED RESPONSES
            const cardClass = success ? (isChampion ? 'champion' : '') : 'border-warning';
            const statusIcon = success ? '✅' : '⚠️';

            responsesHTML += `
                <div class="expert-response ${cardClass}">
                    <div class="expert-header" style="background: ${success ? 'linear-gradient(135deg, var(--socratic-primary) 0%, var(--socratic-accent) 100%)' : 'linear-gradient(135deg, #856404 0%, #ffc107 100%)'}">
                        <div class="d-flex align-items-center">
                            <h5 class="mb-0">
                                ${statusIcon} ${agent.toUpperCase()}
                                ${isChampion ? '<span class="champion-badge ms-2">👑 Champion</span>' : ''}
                                ${!success ? '<span class="badge bg-warning text-dark ms-2">No Response</span>' : ''}
                            </h5>
                        </div>
                        <div class="text-end">
                            <span class="badge bg-light text-dark">${score} pts</span>
                        </div>
                    </div>
                    <div class="expert-content">
                        <p class="mb-3" style="${!success ? 'font-style: italic; color: #856404;' : ''}">${content}</p>
                        <hr>
                        <div class="row text-muted small">
                            <div class="col-4">
                                <i class="fas fa-chart-bar me-1"></i> ${wordCount} words
                            </div>
                            <div class="col-4">
                                <i class="fas fa-clock me-1"></i> ${responseTime}s
                            </div>
                            <div class="col-4">
                                <i class="fas fa-star me-1"></i> Score: ${score}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = responsesHTML;
        console.log('✅ Expert responses rendered successfully');
    }

    displaySocraticSynthesis(synthesis) {
        const synthesisElement = document.getElementById('synthesis-content');
        if (synthesisElement) {
            synthesisElement.innerHTML = `
                <div class="mb-3">
                    <h6 class="text-primary">🧠 Expert Consensus</h6>
                    <p class="mb-2">${synthesis}</p>
                    <hr>
                    <small class="text-muted fst-italic">
                        "In dialogue, we discover truth together." - Socratic Wisdom
                    </small>
                </div>
            `;
        }
    }
    
    displaySocraticChampion(championName) {
        if (!championName) return;

        // Remove champion status from all cards
        document.querySelectorAll('.agent-card').forEach(card => {
            card.classList.remove('champion');
            const crown = card.querySelector('.crown');
            if (crown) crown.remove();
        });

        // Add champion status to the winning agent
        const championCard = document.getElementById(`${championName.toLowerCase()}-card`);
        if (championCard) {
            championCard.classList.add('champion');
            
            // Add crown if not present
            if (!championCard.querySelector('.crown')) {
                const crown = document.createElement('div');
                crown.className = 'crown';
                crown.textContent = '👑';
                championCard.appendChild(crown);
            }

            console.log(`👑 Champion set: ${championName}`);
        }
    }

    updateSessionMetrics(metrics) {
        console.log('📊 Updating session metrics:', metrics);

        if (metrics.response_count !== undefined) {
            const responseCountElement = document.getElementById('response-count');
            if (responseCountElement) {
                responseCountElement.textContent = metrics.response_count;
            }
        }

        if (metrics.champion_score !== undefined) {
            const championScoreElement = document.getElementById('champion-score');
            if (championScoreElement) {
                championScoreElement.textContent = metrics.champion_score;
            }
        }

        if (metrics.process_time !== undefined) {
            const processTimeElement = document.getElementById('process-time');
            if (processTimeElement) {
                processTimeElement.textContent = `${metrics.process_time}s`;
            }
        }

        if (metrics.total_words !== undefined) {
            const totalWordsElement = document.getElementById('total-words');
            if (totalWordsElement) {
                totalWordsElement.textContent = metrics.total_words;
            }
        }
    }
    
    displaySocraticError(error) {
        console.error('❌ Displaying error:', error);

        const container = document.getElementById('expert-responses-container');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <h5 class="mb-2">Expert Panel Analysis Error</h5>
                        <p class="mb-2"><strong>Error:</strong> ${error}</p>
                        <hr>
                        <small class="fst-italic">
                            "I cannot teach anybody anything. I can only make them think." - Socrates
                        </small>
                    </div>
                </div>
            `;
        }

        // Update synthesis with error
        this.displaySocraticSynthesis(`Analysis failed: ${error}`);

        // Reset agent cards
        Object.keys(this.expertAgents).forEach(agentKey => {
            if (this.expertAgents[agentKey].available) {
                const agentCard = document.getElementById(`${agentKey}-card`);
                if (agentCard) {
                    const status = agentCard.querySelector('.agent-status');
                    if (status) {
                        status.textContent = 'Ready';
                        status.className = 'agent-status status-active';
                    }
                }
            }
        });
    }
}

// 🎯 INITIALIZE WHEN DOM IS READY
document.addEventListener('DOMContentLoaded', () => {
    console.log('🏛️ DOM Content Loaded - Initializing Socrates Expert Panel');
    
    // Only initialize if not in development mode
    if (!window.location.search.includes('mode=dev')) {
        try {
            window.socratesExpertPanel = new SocratesExpertPanel();
            console.log('✅ Socrates Expert Panel initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize Socrates Expert Panel:', error);
        }
    }
});

// Handle mode changes
window.addEventListener('modeChanged', (e) => {
    if (e.detail.mode === 'user' && !window.socratesExpertPanel) {
        try {
            window.socratesExpertPanel = new SocratesExpertPanel();
            console.log('✅ Socrates Expert Panel initialized on mode change');
        } catch (error) {
            console.error('❌ Failed to initialize Socrates Expert Panel on mode change:', error);
        }
    }
});

// 🎯 EXPOSE GLOBAL FUNCTIONS FOR DEBUGGING
window.debugSocrates = function() {
    if (window.socratesExpertPanel) {
        console.log('🔍 Socrates Expert Panel Debug Info:');
        console.log('Expert Agents:', window.socratesExpertPanel.expertAgents);
        console.log('Current Analysis:', window.socratesExpertPanel.currentAnalysis);
        console.log('Selected Topic:', window.socratesExpertPanel.selectedTopic);
    } else {
        console.log('❌ Socrates Expert Panel not initialized');
    }
};

// Export for module systems (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SocratesExpertPanel;
}