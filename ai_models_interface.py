"""
Unified interface for multiple AI models to generate research ideas based on proposals.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# AI Model imports
import openai
import google.generativeai as genai
import anthropic
import groq
from dashscope import Generation

# Import prompt templates
from prompt_templates import PromptManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIResponse:
    """Container for AI model responses"""
    model_name: str
    session_id: str
    generated_ideas: str
    timestamp: str
    metadata: Dict[str, Any]

class AIModelsInterface:
    """Unified interface for multiple AI models"""
    
    def __init__(self, config_path: str = "config.env", prompt_template: str = "standard_extension"):
        """Initialize the interface with API keys from config file"""
        self.load_config(config_path)
        self.setup_models()
        self.prompt_manager = PromptManager()
        self.current_template = prompt_template
        
    def load_config(self, config_path: str):
        """Load API keys from environment or config file"""
        # Try to load from environment first
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.google_key = os.getenv('GOOGLE_API_KEY')
        # self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        # self.groq_key = os.getenv('GROQ_API_KEY')
        # self.dashscope_key = os.getenv('DASHSCOPE_API_KEY')
        
        # Load from config file if environment variables not set
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        # Remove quotes if present
                        value = value.strip().strip('"').strip("'")
                        if key == 'OPENAI_API_KEY' and not self.openai_key:
                            self.openai_key = value
                        elif key == 'GOOGLE_API_KEY' and not self.google_key:
                            self.google_key = value
                        # elif key == 'ANTHROPIC_API_KEY' and not self.anthropic_key:
                        #     self.anthropic_key = value
                        # elif key == 'GROQ_API_KEY' and not self.groq_key:
                        #     self.groq_key = value
                        # elif key == 'DASHSCOPE_API_KEY' and not self.dashscope_key:
                        #     self.dashscope_key = value
    
    def setup_models(self):
        """Initialize AI model clients"""
        self.models = {}
        
        # OpenAI (GPT)
        if self.openai_key:
            openai.api_key = self.openai_key
            self.models['gpt-4'] = self._call_openai
            logger.info("OpenAI GPT-4 initialized")
        
        # Google Gemini (temporarily disabled due to model name issues)
        # if self.google_key:
        #     genai.configure(api_key=self.google_key)
        #     self.models['gemini-1.5-flash-002'] = self._call_gemini
        #     logger.info("Google Gemini initialized")
        
        # Anthropic Claude
        # if self.anthropic_key:
        #     self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        #     self.models['claude-3'] = self._call_claude
        #     logger.info("Anthropic Claude initialized")
        
        # # Grok
        # if self.groq_key:
        #     self.groq_client = groq.Groq(api_key=self.groq_key)
        #     self.models['grok'] = self._call_groq
        #     logger.info("Grok initialized")
        
        # # Qwen (via DashScope)
        # if self.dashscope_key:
        #     self.models['qwen'] = self._call_qwen
        #     logger.info("Qwen initialized")
    
    def _call_openai(self, prompt: str, **kwargs) -> str:
        """Call OpenAI GPT-4"""
        try:
            client = openai.OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0),
                max_tokens=kwargs.get('max_tokens', 5000)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"Error: {str(e)}"
    
    def _call_gemini(self, prompt: str, **kwargs) -> str:
        """Call Google Gemini with fallback models"""
        # Try different Gemini models in order of preference
        models_to_try = ['gemini-1.5-flash-002', 'gemini-1.5-pro-002', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying Gemini model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=kwargs.get('temperature', 0),
                        max_output_tokens=kwargs.get('max_tokens', 5000)
                    )
                )
                
                # Check if response is valid
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.finish_reason == 1:  # STOP - normal completion
                        logger.info(f"Successfully used {model_name}")
                        return response.text
                    elif candidate.finish_reason == 2:  # MAX_TOKENS
                        logger.warning(f"{model_name} response truncated due to max tokens")
                        return response.text
                    elif candidate.finish_reason == 3:  # SAFETY
                        logger.warning(f"{model_name} response blocked by safety filters, trying next model")
                        continue
                    elif candidate.finish_reason == 4:  # RECITATION
                        logger.warning(f"{model_name} response blocked due to recitation, trying next model")
                        continue
                    else:
                        logger.warning(f"{model_name} response has finish_reason: {candidate.finish_reason}")
                        if response.text:
                            return response.text
                        else:
                            continue
                else:
                    logger.warning(f"No candidates in {model_name} response, trying next model")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error with {model_name}: {e}, trying next model")
                continue
        
        # If all models failed
        logger.error("All Gemini models failed")
        return "Error: All Gemini models failed to generate response"
    
    # def _call_claude(self, prompt: str, **kwargs) -> str:
    #     """Call Anthropic Claude"""
    #     try:
    #         response = self.anthropic_client.messages.create(
    #             model="claude-3-sonnet-20240229",
    #             max_tokens=kwargs.get('max_tokens', 2000),
    #             temperature=kwargs.get('temperature', 0.7),
    #             messages=[{"role": "user", "content": prompt}]
    #         )
    #         return response.content[0].text
    #     except Exception as e:
    #         logger.error(f"Claude error: {e}")
    #         return f"Error: {str(e)}"
    
    # def _call_groq(self, prompt: str, **kwargs) -> str:
    #     """Call Grok"""
    #     try:
    #         response = self.groq_client.chat.completions.create(
    #             model="llama3-8b-8192",
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=kwargs.get('temperature', 0.7),
    #             max_tokens=kwargs.get('max_tokens', 2000)
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         logger.error(f"Grok error: {e}")
    #         return f"Error: {str(e)}"
    
    # def _call_qwen(self, prompt: str, **kwargs) -> str:
    #     """Call Qwen via DashScope"""
    #     try:
    #         response = Generation.call(
    #             model='qwen-turbo',
    #             prompt=prompt,
    #             api_key=self.dashscope_key,
    #             temperature=kwargs.get('temperature', 0.7),
    #             max_tokens=kwargs.get('max_tokens', 2000)
    #         )
    #         return response.output.text
    #     except Exception as e:
    #         logger.error(f"Qwen error: {e}")
    #         return f"Error: {str(e)}"
    
    def generate_research_ideas(self, research_call: str, model_name: str, 
                               prompt_template: str = None, **kwargs) -> AIResponse:
        """Generate research ideas based on the research call using a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        # Use specified template or current default
        template_name = prompt_template or self.current_template
        
        # Create the prompt using the specified template with research call
        prompt = self.prompt_manager.format_prompt(template_name, {'research_call': research_call})
        
        # Call the model
        raw_response = self.models[model_name](prompt, **kwargs)
        
        # Try to parse JSON response
        parsed_ideas = self._parse_json_response(raw_response)
        
        # Create response object
        response = AIResponse(
            model_name=model_name,
            session_id='research_call_session',  # Since we're working with research call
            generated_ideas=parsed_ideas,
            timestamp=datetime.now().isoformat(),
            metadata={
                'temperature': kwargs.get('temperature', 0),
                'max_tokens': kwargs.get('max_tokens', 5000),
                'prompt_template': template_name,
                'research_call': research_call,
                'raw_response': raw_response,
                'parsed_successfully': parsed_ideas is not None
            }
        )
        
        return response
    
    def _parse_json_response(self, raw_response: str) -> str:
        """Parse JSON response from AI model, return structured data or raw response if parsing fails"""
        try:
            # Try to extract JSON from the response
            response_text = raw_response.strip()
            
            # Look for JSON object in the response
            if response_text.startswith('{') and response_text.endswith('}'):
                parsed_data = json.loads(response_text)
                return json.dumps(parsed_data, indent=2, ensure_ascii=False)
            else:
                # Try to find JSON within the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx+1]
                    parsed_data = json.loads(json_str)
                    return json.dumps(parsed_data, indent=2, ensure_ascii=False)
                else:
                    # Return raw response if no JSON found
                    logger.warning("No valid JSON found in response, returning raw text")
                    return raw_response
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return raw_response
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return raw_response
    
    async def generate_ideas_for_all_models(self, research_call: str, **kwargs) -> List[AIResponse]:
        """Generate research ideas using all available models"""
        tasks = []
        for model_name in self.models.keys():
            task = asyncio.create_task(
                self._async_generate_ideas(research_call, model_name, **kwargs)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid responses
        valid_responses = []
        for response in responses:
            if isinstance(response, AIResponse):
                valid_responses.append(response)
            else:
                logger.error(f"Error generating response: {response}")
        
        return valid_responses
    
    async def _async_generate_ideas(self, research_call: str, model_name: str, **kwargs) -> AIResponse:
        """Async wrapper for idea generation"""
        return self.generate_research_ideas(research_call, model_name, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_available_templates(self) -> List[Dict[str, str]]:
        """Get list of available prompt templates"""
        return self.prompt_manager.list_templates()
    
    def set_prompt_template(self, template_name: str):
        """Set the default prompt template"""
        if template_name not in self.prompt_manager.get_all_templates():
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.prompt_manager.get_all_templates().keys())}")
        self.current_template = template_name
        logger.info(f"Set prompt template to: {template_name}")
    
    def compare_templates(self, proposal: Dict[str, Any], template_names: List[str] = None) -> Dict[str, str]:
        """Compare different prompt templates for the same proposal"""
        if template_names is None:
            template_names = list(self.prompt_manager.get_all_templates().keys())
        
        return self.prompt_manager.compare_templates(template_names, proposal)
    
    async def generate_ideas_with_multiple_templates(self, proposal: Dict[str, Any], 
                                                   model_name: str, 
                                                   template_names: List[str] = None,
                                                   **kwargs) -> List[AIResponse]:
        """Generate ideas using multiple prompt templates for comparison"""
        if template_names is None:
            template_names = list(self.prompt_manager.get_all_templates().keys())
        
        responses = []
        for template_name in template_names:
            try:
                response = self.generate_research_ideas(
                    proposal, model_name, prompt_template=template_name, **kwargs
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Error with template {template_name}: {e}")
        
        return responses
