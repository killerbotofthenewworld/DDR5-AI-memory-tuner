"""
Optional LLM Integration for DDR5 optimization
Lightweight implementation with multiple provider support
"""
import json
import asyncio
from typing import Dict, List, Any, Optional
import streamlit as st
from dataclasses import dataclass
import requests


@dataclass
class LLMConfig:
    """LLM configuration settings"""
    provider: str = "none"  # none, openai, anthropic, ollama, local
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    enabled: bool = False


class DDR5LLMAssistant:
    """Optional LLM assistant for DDR5 optimization guidance"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.conversation_history = []
        
    def is_available(self) -> bool:
        """Check if LLM integration is available and configured"""
        return (self.config.enabled and 
                self.config.provider != "none" and
                (self.config.api_key or self.config.provider in ["ollama", "local"]))
    
    async def get_optimization_advice(self, 
                                    current_config: Dict[str, Any],
                                    performance_data: Dict[str, Any],
                                    goal: str = "balanced") -> Optional[str]:
        """Get LLM advice for DDR5 optimization"""
        
        if not self.is_available():
            return None
            
        try:
            prompt = self._create_optimization_prompt(current_config, performance_data, goal)
            response = await self._query_llm(prompt)
            return response
            
        except Exception as e:
            st.error(f"LLM Assistant unavailable: {e}")
            return None
    
    async def explain_configuration(self, config: Dict[str, Any]) -> Optional[str]:
        """Get LLM explanation of a DDR5 configuration"""
        
        if not self.is_available():
            return None
            
        prompt = f"""
        Explain this DDR5 memory configuration in simple terms:
        
        Frequency: {config.get('frequency', 'Unknown')} MT/s
        Timings: CL{config.get('cl', '?')}-{config.get('trcd', '?')}-{config.get('trp', '?')}-{config.get('tras', '?')}
        Voltage: VDDQ {config.get('vddq', '?')}V, VPP {config.get('vpp', '?')}V
        
        Provide a brief, user-friendly explanation of what these settings mean and their impact on performance.
        Keep it under 150 words.
        """
        
        try:
            return await self._query_llm(prompt)
        except:
            return None
    
    async def troubleshoot_issues(self, issues: List[str]) -> Optional[str]:
        """Get LLM help with troubleshooting DDR5 issues"""
        
        if not self.is_available():
            return None
            
        issues_text = "\\n".join([f"- {issue}" for issue in issues])
        prompt = f"""
        Help troubleshoot these DDR5 memory issues:
        
        {issues_text}
        
        Provide practical troubleshooting steps in order of priority.
        Focus on safe, commonly effective solutions.
        Keep response under 200 words.
        """
        
        try:
            return await self._query_llm(prompt)
        except:
            return None
    
    def _create_optimization_prompt(self, 
                                  config: Dict[str, Any],
                                  performance: Dict[str, Any], 
                                  goal: str) -> str:
        """Create optimization prompt for LLM"""
        
        return f"""
        You are a DDR5 memory optimization expert. Analyze this configuration and suggest improvements.
        
        Current Configuration:
        - Frequency: {config.get('frequency', 'Unknown')} MT/s
        - Timings: CL{config.get('cl', '?')}-{config.get('trcd', '?')}-{config.get('trp', '?')}-{config.get('tras', '?')}
        - Voltages: VDDQ {config.get('vddq', '?')}V, VPP {config.get('vpp', '?')}V
        
        Current Performance:
        - Bandwidth: {performance.get('bandwidth_gbps', 'Unknown')} GB/s
        - Latency: {performance.get('latency_ns', 'Unknown')} ns
        - Stability: {performance.get('stability_score', 'Unknown')}%
        
        Optimization Goal: {goal}
        
        Provide 2-3 specific, actionable recommendations to improve this configuration.
        Consider JEDEC compliance and safety. Keep response under 300 words.
        """
    
    async def _query_llm(self, prompt: str) -> str:
        """Query the configured LLM provider"""
        
        if self.config.provider == "openai":
            return await self._query_openai(prompt)
        elif self.config.provider == "anthropic":
            return await self._query_anthropic(prompt)
        elif self.config.provider == "ollama":
            return await self._query_ollama(prompt)
        elif self.config.provider == "local":
            return await self._query_local(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a DDR5 memory optimization expert."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API error: {response.status_code}")
    
    async def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude API"""
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            raise Exception(f"Anthropic API error: {response.status_code}")
    
    async def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama instance"""
        data = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama error: {response.status_code}")
    
    async def _query_local(self, prompt: str) -> str:
        """Query local LLM endpoint"""
        try:
            # Support for various local LLM servers
            endpoint = self.config.api_key or "http://localhost:8080/v1/completions"
            
            data = {
                "model": self.config.model,
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            response = requests.post(
                endpoint,
                json=data,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                # Handle different response formats
                if "choices" in result:
                    return result["choices"][0]["text"]
                elif "response" in result:
                    return result["response"]
                elif "content" in result:
                    return result["content"]
                else:
                    return str(result)
            else:
                raise Exception(f"Local LLM error: {response.status_code} - {response.text}")
                
        except Exception as e:
            return f"Local LLM unavailable: {e}. Try starting your local LLM server or use Ollama."


class LLMInterface:
    """Streamlit interface for LLM configuration and interaction"""
    
    def __init__(self):
        self.assistant = None
        self._load_config()
    
    def _load_config(self):
        """Load LLM configuration from session state"""
        config = LLMConfig(
            provider=st.session_state.get('llm_provider', 'none'),
            api_key=st.session_state.get('llm_api_key'),
            model=st.session_state.get('llm_model', 'gpt-3.5-turbo'),
            enabled=st.session_state.get('llm_enabled', False)
        )
        self.assistant = DDR5LLMAssistant(config)
    
    def render_configuration_panel(self):
        """Render LLM configuration panel"""
        with st.expander("🤖 AI Assistant Configuration (Optional)", expanded=False):
            st.info("Configure an AI assistant for natural language DDR5 optimization help.")
            
            # Enable/disable toggle
            enabled = st.checkbox(
                "Enable AI Assistant",
                value=st.session_state.get('llm_enabled', False),
                help="Enable natural language AI assistance for DDR5 optimization"
            )
            
            if enabled:
                # Provider selection
                provider = st.selectbox(
                    "AI Provider",
                    options=['none', 'openai', 'anthropic', 'ollama', 'local'],
                    index=0,
                    help="Choose your AI provider. Ollama and Local are free options."
                )
                
                if provider != 'none':
                    # Model selection
                    models = {
                        'openai': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'],
                        'anthropic': ['claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'],
                        'ollama': [
                            'llama3.2:3b', 'llama3.2:1b', 'llama3.1:8b', 'llama3.1:70b',
                            'llama2:7b', 'llama2:13b', 'llama2:70b',
                            'codellama:7b', 'codellama:13b', 'codellama:34b',
                            'mistral:7b', 'mistral:instruct',
                            'phi3:mini', 'phi3:medium',
                            'gemma:2b', 'gemma:7b',
                            'qwen2:0.5b', 'qwen2:1.5b', 'qwen2:7b',
                            'neural-chat:7b'
                        ],
                        'local': [
                            'custom-llama', 'custom-mistral', 'custom-phi3',
                            'huggingface-local', 'transformers-local'
                        ]
                    }
                    
                    model = st.selectbox(
                        "Model",
                        options=models.get(provider, []),
                        help=f"Choose the {provider} model to use"
                    )
                    
                    # API key or endpoint configuration
                    if provider in ['openai', 'anthropic']:
                        api_key = st.text_input(
                            "API Key",
                            type="password",
                            help=f"Enter your {provider} API key"
                        )
                    elif provider == 'ollama':
                        api_key = None
                        st.info("💡 Make sure Ollama is running: `ollama serve`")
                        st.info("📥 Pull models with: `ollama pull llama3.2:3b`")
                    elif provider == 'local':
                        api_key = st.text_input(
                            "Local Endpoint URL",
                            value="http://localhost:8080/v1/completions",
                            help="URL for your local LLM server"
                        )
                    else:
                        api_key = None
                    
                    # Save configuration
                    if st.button("Save AI Configuration"):
                        st.session_state['llm_enabled'] = enabled
                        st.session_state['llm_provider'] = provider
                        st.session_state['llm_model'] = model
                        st.session_state['llm_api_key'] = api_key
                        self._load_config()
                        st.success("AI Assistant configuration saved!")
                        
                        # Test connection
                        if self.assistant.is_available():
                            st.success("✅ AI Assistant is ready!")
                        else:
                            st.warning("⚠️ AI Assistant configuration incomplete")
    
    def render_chat_interface(self):
        """Render AI chat interface"""
        if not self.assistant or not self.assistant.is_available():
            st.info("Configure AI Assistant above to enable natural language help.")
            return
        
        st.subheader("🤖 DDR5 AI Assistant")
        
        # Chat input
        user_question = st.text_input(
            "Ask about DDR5 optimization:",
            placeholder="e.g., 'How can I improve gaming performance?' or 'Explain these timings'"
        )
        
        if user_question and st.button("Ask Assistant"):
            with st.spinner("AI Assistant is thinking..."):
                try:
                    # Simple keyword-based routing
                    if any(word in user_question.lower() for word in ['explain', 'what', 'meaning']):
                        # Configuration explanation
                        config = st.session_state.get('current_config', {})
                        response = asyncio.run(self.assistant.explain_configuration(config))
                    else:
                        # General optimization advice
                        config = st.session_state.get('current_config', {})
                        performance = st.session_state.get('current_performance', {})
                        response = asyncio.run(self.assistant.get_optimization_advice(
                            config, performance, "balanced"
                        ))
                    
                    if response:
                        st.success("🤖 AI Assistant Response:")
                        st.write(response)
                    else:
                        st.error("AI Assistant is temporarily unavailable.")
                        
                except Exception as e:
                    st.error(f"AI Assistant error: {e}")


def create_llm_interface():
    """Create and return LLM interface instance"""
    return LLMInterface()
