"""
Prompt templates for different research idea generation strategies.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Container for prompt template information"""
    name: str
    description: str
    template: str
    parameters: List[str] = None

class PromptManager:
    """Manager for different prompt templates"""
    
    def __init__(self):
        """Initialize with default prompt templates"""
        self.templates = self._create_default_templates()
    
    def _create_default_templates(self) -> Dict[str, PromptTemplate]:
        """Create default prompt templates"""
        templates = {}
        
        # Template 1: Single Scientist Approach
        templates['single_scientist'] = PromptTemplate(
            name="Single Scientist Approach",
            description="Generate research ideas that are most innovative, critical, and aligned with the provided research call",
            template="""
You are a scientist who is the best in the field of the research call in the world, you are tasked with generating innovative research ideas based on the following research call.

RESEARCH CALL:
{research_call}

TASK:
Based on this research call, generate the most innovative and impactful 10 research ideas that address the goals 
and objectives outlined in the call and align with the funding organization's mission

For each idea, provide:
- A clear, concise title
- An abstract (250-500 words)

IMPORTANT: Format your response as a valid JSON object with the following structure:
{{
  "research_ideas": [
    {{
      "title": "Research Idea Title",
      "abstract": "Detailed abstract of the research idea..."
    }},
    {{
      "title": "Another Research Idea Title", 
      "abstract": "Detailed abstract of this research idea..."
    }}
  ]
}}

Ensure the JSON is properly formatted and valid.
""",
            parameters=['research_call']
        )
        
        # Template 2: Groups of Scientists Approach
        templates['groups_of_scientists'] = PromptTemplate(
            name="Groups of Scientists Approach",
            description="Generate ideas by critically analyzing the research call",
            template="""
You are 10 of the best scientists in the world in the field of the research call, tasked with generating innovative research ideas based on the following research call.

RESEARCH CALL:
{research_call}

TASK:
Based on this research call, generate the most innovative and impactful 10 research ideas that address the goals 
and objectives outlined in the call and align with the funding organization's mission

For each idea, provide:
- A clear, concise title
- An abstract (250-500 words)

IMPORTANT: Format your response as a valid JSON object with the following structure:
{{
  "research_ideas": [
    {{
      "title": "Research Idea Title",
      "abstract": "Detailed abstract of the research idea..."
    }},
    {{
      "title": "Another Research Idea Title", 
      "abstract": "Detailed abstract of this research idea..."
    }}
  ]
}}

Ensure the JSON is properly formatted and valid.
""",
            parameters=['research_call']
        )

        # Template 3: Groups of Interdisciplinary Scientists Approach
        templates['groups_of_interdisciplinary_scientists'] = PromptTemplate(
            name="Groups of Interdisciplinary Scientists Approach",
            description="Generate ideas by critically analyzing the research call",
            template="""
You are 10 of the best scientists in the world each in different fields of expertise (biology, chemistry, physics, medicine, computer science, etc), 
you are tasked with incorporating your diverse expertise and generating innovative research ideas based on the following research call.

RESEARCH CALL:
{research_call}

TASK:
Based on this research call, generate the most innovative and impactful 10 research ideas that address the goals 
and objectives outlined in the call and align with the funding organization's mission

For each idea, provide:
- A clear, concise title
- An abstract (between 250-500 words)

IMPORTANT: Format your response as a valid JSON object with the following structure:
{{
  "research_ideas": [
    {{
      "title": "Research Idea Title",
      "abstract": "Detailed abstract of the research idea..."
    }},
    {{
      "title": "Another Research Idea Title", 
      "abstract": "Detailed abstract of this research idea..."
    }}
  ]
}}

Ensure the JSON is properly formatted and valid.
""",
            parameters=['research_call']
        )
        
       
        
        return templates
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[template_name]
    
    def get_all_templates(self) -> Dict[str, PromptTemplate]:
        """Get all available templates"""
        return self.templates
    
    def list_templates(self) -> List[Dict[str, str]]:
        """Get a list of all templates with their names and descriptions"""
        return [
            {
                'name': name,
                'description': template.description,
                'parameters': template.parameters
            }
            for name, template in self.templates.items()
        ]
    
    def create_custom_template(self, name: str, description: str, template: str, parameters: List[str] = None):
        """Create a custom template"""
        self.templates[name] = PromptTemplate(
            name=name,
            description=description,
            template=template,
            parameters=parameters or []
        )
    
    def format_prompt(self, template_name: str, data: Dict[str, Any]) -> str:
        """Format a prompt using a specific template and data"""
        template = self.get_template(template_name)
        
        # Prepare data for formatting
        format_data = {
            'research_call': data.get('research_call', 'N/A')
        }
        
        try:
            return template.template.format(**format_data)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for template '{template_name}': {e}")
    
    def compare_templates(self, template_names: List[str], data: Dict[str, Any]) -> Dict[str, str]:
        """Compare multiple templates for the same data"""
        results = {}
        for template_name in template_names:
            try:
                results[template_name] = self.format_prompt(template_name, data)
            except Exception as e:
                results[template_name] = f"Error: {str(e)}"
        return results

# Example usage and testing
if __name__ == "__main__":
    # Test the prompt manager
    manager = PromptManager()
    
    # List all templates
    print("Available templates:")
    for template_info in manager.list_templates():
        print(f"- {template_info['name']}: {template_info['description']}")
    
    # Example research call
    example_research_call = {
        "research_call": "We seek innovative research proposals that advance our understanding of artificial intelligence applications in healthcare, with particular focus on improving patient outcomes and reducing healthcare disparities."
    }
    
    # Test different templates
    print("\nTesting templates:")
    for template_name in manager.get_all_templates().keys():
        try:
            prompt = manager.format_prompt(template_name, example_research_call)
            print(f"\n{template_name}:")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        except Exception as e:
            print(f"\n{template_name}: Error - {e}")
