"""
Prompt templates for different research idea generation strategies.
Consolidated approach with configurable roles.
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
        
        # Define role configurations
        self.role_configs = {
            'single_scientist': {
                'role_description': 'a scientist who is the best in the field of the research call in the world',
                'description': 'Individual expert perspective with focused expertise',
                'perspective': 'individual expert',
                'approach': 'focused expertise'
            },
            'groups_of_scientists': {
                'role_description': '10 of the best scientists in the world all in the same field of the research call',
                'description': 'Collaborative team perspective with collective expertise',
                'perspective': 'collaborative team',
                'approach': 'collective expertise'
            },
            'groups_of_interdisciplinary_scientists': {
                'role_description': '10 of the best scientists in the world each in different fields of expertise (biology, chemistry, physics, medicine, computer science, etc)',
                'description': 'Diverse interdisciplinary team with cross-field integration',
                'perspective': 'diverse interdisciplinary team',
                'approach': 'cross-field integration'
            }
        }
    
    def _create_default_templates(self) -> Dict[str, PromptTemplate]:
        """Create default prompt templates"""
        templates = {}
        
        # Template 1: Research Ideas Generation (Dynamic Role)
        templates['generate_ideas'] = PromptTemplate(
            name="Generate Research Ideas",
            description="Generate innovative research ideas with configurable role perspective",
            template="""
You are {role_description}, you are tasked with generating innovative research ideas based on the following research call.

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
            parameters=['research_call', 'role_description']
        )

        # Template 2: Research Proposals Generation (Dynamic Role)
        templates['generate_proposals'] = PromptTemplate(
            name="Generate Research Proposals",
            description="Generate comprehensive research proposals with configurable role perspective",
            template="""
You are {role_description} writing a comprehensive research proposal. Based on the provided research idea, create a full proposal with the following sections:

RESEARCH IDEA:
Title: {title}
Abstract: {abstract}

TASK:
Generate a comprehensive research proposal that includes:

1. BACKGROUND AND SIGNIFICANCE (300-400 words)
   - Context and current state of the field
   - Key gaps and limitations in current knowledge
   - Why this research is important and timely

2. RESEARCH QUESTIONS AND HYPOTHESES (200-300 words)
   - Specific research questions to be addressed
   - Testable hypotheses
   - Expected outcomes

3. METHODS AND APPROACH (400-500 words)
   - Data sources and datasets to be used
   - Analytical methods and computational approaches
   - Experimental design (if applicable)
   - Timeline and milestones

4. EXPECTED OUTCOMES AND IMPACT (200-300 words)
   - Intended contributions to the field
   - Broader impacts and applications
   - Potential for follow-up research

IMPORTANT: Format your response as a valid JSON object with the following structure:
{{
  "proposal": {{
    "title": {title},
    "abstract": {abstract},
    "background_and_significance": "Background section...",
    "research_questions_and_hypotheses": "Research questions section...",
    "methods_and_approach": "Methods section...",
    "expected_outcomes_and_impact": "Outcomes section..."
  }}
}}

Ensure the JSON is properly formatted and valid.
""",
            parameters=['title', 'abstract', 'role_description']
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
    
    def format_prompt(self, template_name: str, data: Dict[str, Any], role: str = 'single_scientist') -> str:
        """Format a prompt using a specific template, data, and role"""
        template = self.get_template(template_name)
        
        # Get role description
        if role not in self.role_configs:
            raise ValueError(f"Role '{role}' not found. Available: {list(self.role_configs.keys())}")
        
        role_description = self.role_configs[role]['role_description']
        
        # Prepare data for formatting
        format_data = {
            'research_call': data.get('research_call', 'N/A'),
            'title': data.get('title', 'N/A'),
            'abstract': data.get('abstract', 'N/A'),
            'role_description': role_description
        }
        
        try:
            return template.template.format(**format_data)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for template '{template_name}': {e}")
    
    def compare_templates(self, template_names: List[str], data: Dict[str, Any], role: str = 'single_scientist') -> Dict[str, str]:
        """Compare multiple templates for the same data and role"""
        results = {}
        for template_name in template_names:
            try:
                results[template_name] = self.format_prompt(template_name, data, role)
            except Exception as e:
                results[template_name] = f"Error: {str(e)}"
        return results
    
    def get_available_roles(self) -> List[str]:
        """Get list of available roles"""
        return list(self.role_configs.keys())
    
    def get_role_info(self, role: str) -> Dict[str, str]:
        """Get information about a specific role"""
        if role not in self.role_configs:
            raise ValueError(f"Role '{role}' not found. Available: {list(self.role_configs.keys())}")
        return self.role_configs[role]

# Backward compatibility methods for existing scripts
def get_legacy_template_name(role: str, task_type: str) -> str:
    """Convert role + task_type to legacy template names for backward compatibility"""
    if task_type == 'ideas':
        return role
    elif task_type == 'proposals':
        return f"{role}_proposal"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def format_legacy_prompt(template_name: str, data: Dict[str, Any]) -> str:
    """Legacy method for backward compatibility with existing scripts"""
    manager = PromptManager()
    
    # Determine if this is a legacy template name
    if template_name in ['single_scientist', 'groups_of_scientists', 'groups_of_interdisciplinary_scientists']:
        # This is an ideas template
        return manager.format_prompt('generate_ideas', data, template_name)
    elif template_name.endswith('_proposal'):
        # This is a proposal template
        role = template_name.replace('_proposal', '')
        return manager.format_prompt('generate_proposals', data, role)
    else:
        # Try to use as-is (for new template names)
        return manager.format_prompt(template_name, data)

# Example usage and testing
if __name__ == "__main__":
    # Test the prompt manager
    manager = PromptManager()
    
    # List all templates
    print("Available templates:")
    for template_info in manager.list_templates():
        print(f"- {template_info['name']}: {template_info['description']}")
    
    print("\nAvailable roles:")
    for role in manager.get_available_roles():
        role_info = manager.get_role_info(role)
        print(f"- {role}: {role_info['description']}")
    
    # Example research call
    example_data = {
        "research_call": "We seek innovative research proposals that advance our understanding of artificial intelligence applications in healthcare, with particular focus on improving patient outcomes and reducing healthcare disparities.",
        "title": "AI-Powered Diagnostic System",
        "abstract": "An innovative approach to using machine learning for early disease detection..."
    }
    
    # Test different roles with ideas template
    print("\nTesting ideas generation with different roles:")
    for role in manager.get_available_roles():
        try:
            prompt = manager.format_prompt('generate_ideas', example_data, role)
            print(f"\n{role} ideas prompt:")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        except Exception as e:
            print(f"\n{role} ideas: Error - {e}")
    
    # Test different roles with proposals template
    print("\nTesting proposals generation with different roles:")
    for role in manager.get_available_roles():
        try:
            prompt = manager.format_prompt('generate_proposals', example_data, role)
            print(f"\n{role} proposals prompt:")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        except Exception as e:
            print(f"\n{role} proposals: Error - {e}")