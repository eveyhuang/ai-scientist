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

        # Template 1b: Research Ideas Generation (No Role Description)
        templates['generate_ideas_no_role'] = PromptTemplate(
            name="Generate Research Ideas (No Role)",
            description="Generate innovative research ideas without any role perspective",
            template="""
            Generate innovative research ideas based on the following research call.

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

        # Template 2: Research Proposals Generation (Dynamic Role)
        templates['generate_proposals'] = PromptTemplate(
            name="Generate Research Proposals",
            description="Generate comprehensive research proposals with configurable role perspective",
            template="""
            You are {role_description} writing a comprehensive research proposal based on the provided title and abstract, ensuring it aligns with the research call requirements.

            TASK: Generate a research proposal based on the provided research title and abstract.

            RESEARCH CALL:
            {research_call}

            RESEARCH IDEA TO EXPAND INTO A PROPOSAL:
            Title: {title}
            Abstract: {abstract}

            Create a comprehensive research proposal that is to the highest standard for writing quality, concreteness, coherence, logic, and scientific rigor. 
            It should include these section and meet the minimum word counts for each section:

            1. BACKGROUND AND SIGNIFICANCE (600-800 words)
            - Comprehensive context and current state of the field
            - Detailed literature review of relevant work
            - Key gaps and limitations in current knowledge
            - Why this research is important and timely

            2. RESEARCH QUESTIONS AND HYPOTHESES (600-800 words)
            - Specific, detailed research questions to be addressed
            - Testable hypotheses with clear predictions
            - Expected outcomes and deliverables
            - How hypotheses will be tested and validated

            3. METHODS AND APPROACH (800-1000 words)
            - Detailed data sources and datasets to be used
            - Comprehensive analytical methods and computational approaches
            - Experimental design (if applicable) with controls and replicates
            - Timeline and milestones with specific deliverables
            - Statistical analysis plans (if applicable)

            4. EXPECTED OUTCOMES AND IMPACT (600-800 words)
            - Detailed intended contributions to the field
            - Broader impacts and applications
            - Potential for follow-up research and collaborations
            - Dissemination plans and publication strategy
            - Long-term vision and sustainability

            5. BUDGET AND RESOURCES (400-600 words)
            - Detailed budget breakdown by category

            RESPONSE FORMAT: Return ONLY this JSON structure with no additional text:

            {{
            "proposal": {{
                "title": "{title}",
                "abstract": "{abstract}",
                "background_and_significance": "Comprehensive background section...",
                "research_questions_and_hypotheses": "Detailed research questions section...",
                "methods_and_approach": "Comprehensive methods section...",
                "expected_outcomes_and_impact": "Detailed outcomes section...",
                "budget_and_resources": "Detailed budget section..."
            }}
            }}

            IMPORTANT: 
            - Generate ONLY the proposal JSON above
            - Do NOT include research ideas, additional explanations, or any other content
            - Each section must meet the expectation of your role and be substantial, detailed, coherent, logical, and meet the specified word counts.
            - Ensure the JSON is properly formatted and valid
            """,
                        parameters=['title', 'abstract', 'role_description', 'research_call']
        )
        
        # Template 2b: Research Proposals Generation (No Role Description)
        templates['generate_proposals_no_role'] = PromptTemplate(
            name="Generate Research Proposals (No Role)",
            description="Generate comprehensive research proposals without any role perspective",
            template="""
            Write a comprehensive research proposal based on the provided title and abstract, ensuring it aligns with the research call requirements.

            TASK: Generate a research proposal based on the provided research title and abstract.

            RESEARCH CALL:
            {research_call}

            RESEARCH IDEA TO EXPAND INTO A PROPOSAL:
            Title: {title}
            Abstract: {abstract}

            Create a comprehensive research proposal that is to the highest standard for writing quality, concreteness, coherence, logic, and scientific rigor. 
            It should include these section and meet the minimum word counts for each section:

            1. BACKGROUND AND SIGNIFICANCE (600-800 words)
            - Comprehensive context and current state of the field
            - Detailed literature review of relevant work
            - Key gaps and limitations in current knowledge
            - Why this research is important and timely

            2. RESEARCH QUESTIONS AND HYPOTHESES (600-800 words)
            - Specific, detailed research questions to be addressed
            - Testable hypotheses with clear predictions
            - Expected outcomes and deliverables
            - How hypotheses will be tested and validated

            3. METHODS AND APPROACH (800-1000 words)
            - Detailed data sources and datasets to be used
            - Comprehensive analytical methods and computational approaches
            - Experimental design (if applicable) with controls and replicates
            - Timeline and milestones with specific deliverables
            - Statistical analysis plans (if applicable)

            4. EXPECTED OUTCOMES AND IMPACT (600-800 words)
            - Detailed intended contributions to the field
            - Broader impacts and applications
            - Potential for follow-up research and collaborations
            - Dissemination plans and publication strategy
            - Long-term vision and sustainability

            5. BUDGET AND RESOURCES (400-600 words)
            - Detailed budget breakdown by category

            RESPONSE FORMAT: Return ONLY this JSON structure with no additional text:

            {{
            "proposal": {{
                "title": "{title}",
                "abstract": "{abstract}",
                "background_and_significance": "Comprehensive background section...",
                "research_questions_and_hypotheses": "Detailed research questions section...",
                "methods_and_approach": "Comprehensive methods section...",
                "expected_outcomes_and_impact": "Detailed outcomes section...",
                "budget_and_resources": "Detailed budget section..."
            }}
            }}

            IMPORTANT: 
            - Generate ONLY the proposal JSON above
            - Do NOT include research ideas, additional explanations, or any other content
            - Each section must be substantial, detailed, coherent, logical, and meet the specified word counts.
            - Ensure the JSON is properly formatted and valid
            """,
                        parameters=['title', 'abstract', 'research_call']
        )
        
        # Template 3: Comprehensive Evaluation
        templates['eval_comprehensive'] = PromptTemplate(
            name="Comprehensive Evaluation",
            description="Comprehensive evaluation across 10 criteria with structured JSON output",
            template="""You are an expert scientific reviewer evaluating a research proposal for the following funding call:

            {research_call}

            You have been asked to evaluate the following research proposal submitted in response to this call.

            **PROPOSAL TO EVALUATE:**
            ID: {proposal_id}
            Title: {proposal_title}
            Abstract: {proposal_abstract}
            Full Proposal: {proposal_full}

            Your task as a {role_description} is to provide a comprehensive evaluation of this proposal on the following criteria:

            1. **Scientific Merit**: How well does the proposal address a compelling scientific question?
            2. **Alignment with Call**: How well does the proposal align with the funding call requirements?
            3. **Methodology**: Quality and appropriateness of proposed methods
            4. **Innovation**: Novel approaches and contributions
            5. **Feasibility**: Likelihood of successful completion
            6. **Data Synthesis Approach**: Quality of the data synthesis strategy
            7. **Collaborative Approach**: Strength of the team and collaborative plan
            8. **Open Science Commitment**: Adherence to open science principles
            9. **Training Opportunities**: Quality of training plan for trainees
            10. **Impact**: Potential broader impacts and contributions

            IMPORTANT: Return your evaluation as a valid JSON object with the following structure:

            {{
            "evaluation": {{
                "proposal_id": "{proposal_id}",
                "overall_score": <average of all criteria scores, rounded to 1 decimal>,
                "overall_assessment": "<2-3 sentence summary of the proposal's strengths and recommendation>",
                "criteria": [
                {{
                    "criterion": "Scientific Merit",
                    "score": <1-5, where 1=poor, 2=fair, 3=good, 4=very good, 5=excellent>,
                    "justification": "<2-3 sentence explanation for this score>"
                }},
                {{
                    "criterion": "Alignment with Call",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Methodology",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Innovation",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Feasibility",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Data Synthesis Approach",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Collaborative Approach",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Open Science Commitment",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Training Opportunities",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }},
                {{
                    "criterion": "Impact",
                    "score": <1-5>,
                    "justification": "<2-3 sentence explanation>"
                }}
                ]
            }}
            }}

            Provide ONLY the JSON output above with no additional text before or after.""",
                        parameters=['research_call', 'proposal_id', 'proposal_title',  'proposal_abstract', 
                                'proposal_full', 'role_description']
        )
        
        # Template 4: Strengths and Weaknesses Evaluation
        templates['eval_strengths_weaknesses'] = PromptTemplate(
            name="Strengths and Weaknesses Evaluation",
            description="Detailed pros/cons analysis with structured JSON output",
            template="""You are a scientific peer reviewer with the role of {role_description}.

            Research Funding Call:
            {research_call}

            Please review the following proposal:

            **PROPOSAL TO EVALUATE:**
            ID: {proposal_id}
            Title: {proposal_title}
            Abstract: {proposal_abstract}
            Full Proposal: {proposal_full}

            Provide a detailed analysis of this proposal's strengths and weaknesses.

            IMPORTANT: Return your evaluation as a valid JSON object with the following structure:

            {{
            "evaluation": {{
                "proposal_id": "{proposal_id}",
                "overall_score": <1-5, where 1=poor, 2=fair, 3=good, 4=very good, 5=excellent>,
                "recommendation": "<recommend for funding / recommend with revisions / do not recommend>",
                "strengths": [
                "<strength 1: detailed description>",
                "<strength 2: detailed description>",
                "<strength 3: detailed description>",
                "<additional strengths as needed>"
                ],
                "weaknesses": [
                "<weakness 1: detailed description>",
                "<weakness 2: detailed description>",
                "<weakness 3: detailed description>",
                "<additional weaknesses as needed>"
                ],
                "alignment_with_priorities": "<2-3 sentences on how well the proposal addresses funding priorities>",
                "recommendations_for_improvement": [
                "<specific suggestion 1>",
                "<specific suggestion 2>",
                "<specific suggestion 3>",
                "<additional suggestions as needed>"
                ],
                "summary": "<2-3 sentence overall assessment and final recommendation>"
            }}
            }}

            Provide ONLY the JSON output above with no additional text before or after.""",
                        parameters=['research_call', 'proposal_id', 'proposal_title', 'proposal_abstract', 
                                'proposal_full', 'role_description']
        )
        
        # Template 5: Innovation Assessment Evaluation
        templates['eval_innovation_assessment'] = PromptTemplate(
            name="Innovation Assessment Evaluation",
            description="Focus on novelty and innovation with structured JSON output",
            template="""As a {role_description}, evaluate the innovation and novelty of the following research proposal.

            Funding Call Context:
            {research_call}

            **PROPOSAL TO EVALUATE:**
            ID: {proposal_id}
            Title: {proposal_title}
            Abstract: {proposal_abstract}
            Full Proposal: {proposal_full}

            Focus your evaluation on innovation-related criteria.

            IMPORTANT: Return your evaluation as a valid JSON object with the following structure:

            {{
            "evaluation": {{
                "proposal_id": "{proposal_id}",
                "overall_innovation_score": <average of all criteria scores, rounded to 1 decimal>,
                "overall_assessment": "<2-3 sentence summary of the proposal's innovation level>",
                "criteria": [
                {{
                    "criterion": "Novelty of Research Questions",
                    "score": <1-5, where 1=not novel, 2=somewhat novel, 3=novel, 4=highly novel, 5=groundbreaking>,
                    "justification": "<Are the questions truly novel or long-standing puzzles? Explain in 2-3 sentences>"
                }},
                {{
                    "criterion": "Methodological Innovation",
                    "score": <1-5>,
                    "justification": "<Are new analytical strategies or approaches proposed? Explain in 2-3 sentences>"
                }},
                {{
                    "criterion": "Data Integration Innovation",
                    "score": <1-5>,
                    "justification": "<How innovative is the approach to synthesizing existing data? Explain in 2-3 sentences>"
                }},
                {{
                    "criterion": "Potential for New Insights",
                    "score": <1-5>,
                    "justification": "<Likelihood of generating breakthrough insights? Explain in 2-3 sentences>"
                }},
                {{
                    "criterion": "Risk vs Reward Balance",
                    "score": <1-5>,
                    "justification": "<Is there an appropriate balance between ambitious goals and feasibility? Explain in 2-3 sentences>"
                }}
                ]
            }}
            }}

            Provide ONLY the JSON output above with no additional text before or after.""",
                        parameters=['research_call', 'proposal_id', 'proposal_title', 'proposal_abstract', 
                                'proposal_full', 'role_description']
        )
        
        
        # Template 6: Proposal Overlap/Similarity Comparison
        templates['eval_proposal_overlap'] = PromptTemplate(
            name="Proposal Overlap/Similarity Evaluation",
            description="Compare two proposals on overlap dimensions with scoring",
            template="""You are an expert scientific reviewer evaluating the overlap and similarity between two research proposals for the following funding call:

            {research_call}

            You have been asked to assess how similar or overlapping these two proposals are across multiple dimensions.

            **PROPOSAL 1:**
            ID: {proposal_1_id}
            Title: {proposal_1_title}
            Abstract: {proposal_1_abstract}
            Full Proposal: {proposal_1_full}

            **PROPOSAL 2:**
            ID: {proposal_2_id}
            Title: {proposal_2_title}
            Abstract: {proposal_2_abstract}
            Full Proposal: {proposal_2_full}

            Your task is to evaluate the overlap between these two proposals across the following dimensions:

            **D1. Research Question / Aims**
            Are they asking the same "why" or "how" question?
            - 0 = clearly different aims
            - 1 = related theme, different core question
            - 2 = overlapping aim with distinct sub-questions
            - 3 = near-identical aims/hypotheses
            - 4 = identical aims/hypotheses

            **D2. Data / Empirical Context**
            Are they using the same population, dataset, or case study?
            - 0 = different population/dataset/context
            - 1 = same broad domain, different sample
            - 2 = partial overlap (e.g., adjacent cohorts/regions)
            - 3 = same dataset or population with minor differences
            - 4 = same dataset/population/site

            **D3. Methods / Design**
            Are they using the same techniques to get answers?
            - 0 = different methodological families
            - 1 = different designs addressing similar question
            - 2 = same family, different design details
            - 3 = same design and similar analysis plan
            - 4 = same design, measures, instruments, and analysis plan

            **D4. Intended Contribution / Outcomes**
            Do they claim novelty in the same theoretical or practical space?
            - 0 = distinct contributions/literatures
            - 1 = adjacent literatures
            - 2 = same literature, different claimed gap
            - 3 = same gap/contribution
            - 4 = duplicate novelty claim

            **D5. Resources / Timing / Artifacts**
            Do they request similar resources, timelines, or operational plans?
            - 0 = unique partners/resources/timeline
            - 1 = partial overlap
            - 2 = multiple shared resources
            - 3 = same partners/instruments/timeline
            - 4 = effectively the same operational plan

            IMPORTANT: Return your evaluation as a valid JSON object with the following structure:

            {{
            "comparison": {{
                "proposal_1_id": "{proposal_1_id}",
                "proposal_2_id": "{proposal_2_id}",
                "dimensions": [
                {{
                    "dimension": "Research Question / Aims",
                    "score": <0-4>,
                    "justification": "<One sentence explaining the score based on problem framing, aims, or hypotheses>"
                }},
                {{
                    "dimension": "Data / Empirical Context",
                    "score": <0-4>,
                    "justification": "<One sentence explaining the score based on population, dataset, or case study overlap>"
                }},
                {{
                    "dimension": "Methods / Design",
                    "score": <0-4>,
                    "justification": "<One sentence explaining the score based on methodological approach and techniques>"
                }},
                {{
                    "dimension": "Intended Contribution / Outcomes",
                    "score": <0-4>,
                    "justification": "<One sentence explaining the score based on claimed novelty and literature gaps>"
                }},
                {{
                    "dimension": "Resources / Timing / Artifacts",
                    "score": <0-4>,
                    "justification": "<One sentence explaining the score based on resources, timeline, and operational details>"
                }}
                ]
            }}
            }}

            Provide ONLY the JSON output above with no additional text before or after.""",
            parameters=['research_call', 'proposal_1_id', 'proposal_1_title', 'proposal_1_abstract', 
                       'proposal_1_full', 'proposal_2_id', 'proposal_2_title', 'proposal_2_abstract',
                       'proposal_2_full']
        )
        
        # Template 7: Human Criteria Evaluation
        templates['eval_human_criteria'] = PromptTemplate(
            name="Human Criteria Evaluation",
            description="Evaluate proposal based on human reviewer criteria with detailed scoring",
            template="""You are an expert scientific reviewer evaluating a research proposal for the following funding call:

            {research_call}

            You have been asked to evaluate the following research proposal submitted in response to this call.

            **PROPOSAL TO EVALUATE:**
            ID: {proposal_id}
            Title: {proposal_title}
            Abstract: {proposal_abstract}
            Full Proposal: {proposal_full}

            Your task as a {role_description} is to provide a detailed evaluation of this proposal based on the following criteria:

            **EVALUATION CRITERIA:**

            **1. Scientific Merit and Innovation**

            **1a. Relevance to Emergent Phenomena**
            Does the research explicitly address emergent phenomena at the mesoscale in molecular/cellular biosciences?
            - 1 = Not relevant; does not address emergent phenomena
            - 2 = Minimally relevant; tangential connection to emergent phenomena
            - 3 = Moderately relevant; addresses emergent phenomena but not as central focus
            - 4 = Highly relevant; emergent phenomena is a key focus
            - 5 = Exceptionally relevant; directly and explicitly addresses mesoscale emergent phenomena

            **1b. Novelty & Significance**
            Are the questions and approaches innovative? Do they have potential to advance knowledge?
            - 1 = Not novel; incremental work with limited significance
            - 2 = Somewhat novel; modest advancement expected
            - 3 = Novel; clear advancement in the field
            - 4 = Highly novel; significant potential to advance knowledge
            - 5 = Groundbreaking; transformative potential for the field

            **1c. Rigor of Approach**
            Is the proposed methodology clear, logical, and grounded in established or emerging research practices?
            - 1 = Poor; unclear or illogical methodology
            - 2 = Fair; methodology has significant gaps or concerns
            - 3 = Good; solid methodology with minor concerns
            - 4 = Very good; clear, logical, and well-grounded methodology
            - 5 = Excellent; exceptionally rigorous and well-justified approach

            **2. Feasibility**

            **2a. Scope & Timeline**
            Are the goals and milestones realistic for the proposed time frame and planned approach?
            - 1 = Unrealistic; goals are unattainable within proposed timeline
            - 2 = Questionable; significant concerns about feasibility
            - 3 = Reasonable; achievable with noted challenges
            - 4 = Realistic; well-planned scope and timeline
            - 5 = Highly feasible; excellent planning with contingencies

            **3. Data Sources and Limitations**

            **3a. Synthesis Focus**
            Does the proposal clearly demonstrate a synthesis project?
            - 1 = No synthesis; appears to be primarily generating new data
            - 2 = Minimal synthesis; mostly new data generation with some integration
            - 3 = Moderate synthesis; balanced between existing and new data
            - 4 = Strong synthesis; primarily uses existing data with clear integration plan
            - 5 = Exemplary synthesis; exclusively uses existing data with comprehensive integration

            **3b. Data Identification**
            Are the data sources explicitly identified, and are limitations appropriately acknowledged?
            - 1 = Poor; data sources vague and limitations not addressed
            - 2 = Fair; some data sources identified but incomplete or limitations ignored
            - 3 = Good; data sources identified and basic limitations acknowledged
            - 4 = Very good; clear data sources with thoughtful discussion of limitations
            - 5 = Excellent; comprehensive data source specification with thorough limitation analysis

            **4. Open Science Compliance**

            **4a. Open Science Commitment**
            Does the proposal demonstrate a commitment to open, team, and reproducible science principles?
            - 1 = No commitment; does not address open science
            - 2 = Minimal commitment; vague statements without concrete plans
            - 3 = Moderate commitment; some open science practices mentioned
            - 4 = Strong commitment; clear plans for open and reproducible science
            - 5 = Exemplary commitment; comprehensive open science framework with detailed implementation

            IMPORTANT: Return your evaluation as a valid JSON object with the following structure:

            {{
            "evaluation": {{
                "proposal_id": "{proposal_id}",
                "criteria_scores": [
                {{
                    "category": "Scientific Merit and Innovation",
                    "subcriteria": [
                    {{
                        "criterion": "Relevance to Emergent Phenomena",
                        "score": <1-5>,
                        "justification": "<1-2 sentence explanation for this score>"
                    }},
                    {{
                        "criterion": "Novelty & Significance",
                        "score": <1-5>,
                        "justification": "<1-2 sentence explanation for this score>"
                    }},
                    {{
                        "criterion": "Rigor of Approach",
                        "score": <1-5>,
                        "justification": "<1-2 sentence explanation for this score>"
                    }}
                    ],
                    "category_average": <average of subcriteria scores, rounded to 1 decimal>
                }},
                {{
                    "category": "Feasibility",
                    "subcriteria": [
                    {{
                        "criterion": "Scope & Timeline",
                        "score": <1-5>,
                        "justification": "<1-2 sentence explanation for this score>"
                    }}
                    ],
                    "category_average": <score from subcriterion>
                }},
                {{
                    "category": "Data Sources and Limitations",
                    "subcriteria": [
                    {{
                        "criterion": "Synthesis Focus",
                        "score": <1-5>,
                        "justification": "<1-2 sentence explanation for this score>"
                    }},
                    {{
                        "criterion": "Data Identification",
                        "score": <1-5>,
                        "justification": "<1-2 sentence explanation for this score>"
                    }}
                    ],
                    "category_average": <average of subcriteria scores, rounded to 1 decimal>
                }},
                {{
                    "category": "Open Science Compliance",
                    "subcriteria": [
                    {{
                        "criterion": "Open Science Commitment",
                        "score": <1-5>,
                        "justification": "<1-2 sentence explanation for this score>"
                    }}
                    ],
                    "category_average": <score from subcriterion>
                }}
                ],
                "overall_rating": {{
                    "final_numeric_score": <average of all category averages, rounded to 1 decimal>,
                    "narrative_summary": "<One or two paragraphs explaining key strengths and areas for improvement>"
                }}
            }}
            }}

            Provide ONLY the JSON output above with no additional text before or after.""",
            parameters=['research_call', 'proposal_id', 'proposal_title', 'proposal_abstract', 
                       'proposal_full', 'role_description']
        )
        
        # Template 8: Alignment with Call Evaluation
        templates['eval_alignment_with_call'] = PromptTemplate(
            name="Alignment with Call Evaluation",
            description="Evaluate fit with funding requirements with structured JSON output",
            template="""As a {role_description}, assess how well this proposal aligns with the specific requirements of the funding call.

            **FUNDING CALL REQUIREMENTS:**
            {research_call}

            **PROPOSAL TO EVALUATE:**
            ID: {proposal_id}
            Title: {proposal_title}
            Abstract: {proposal_abstract}
            Full Proposal: {proposal_full}

            Evaluate the proposal's alignment with the funding call requirements.

            IMPORTANT: Return your evaluation as a valid JSON object with the following structure:

            {{
            "evaluation": {{
                "proposal_id": "{proposal_id}",
                "overall_alignment_score": <average of all criteria scores, rounded to 1 decimal>,
                "overall_assessment": "<2-3 sentence summary of how well the proposal aligns with the call>",
                "criteria": [
                {{
                    "criterion": "Community-Scale Synthesis",
                    "score": <1-5, where 1=does not meet, 2=partially meets, 3=meets, 4=exceeds, 5=exemplary>,
                    "justification": "<Does it use only existing public data? Provide specific evidence in 2-3 sentences>"
                }},
                {{
                    "criterion": "Collaboration Requirements",
                    "score": <1-5>,
                    "justification": "<Does it require collaboration beyond a single lab? Provide specific evidence in 2-3 sentences>"
                }},
                {{
                    "criterion": "Transdisciplinary Approach",
                    "score": <1-5>,
                    "justification": "<Does it bring together diverse scientific perspectives? Provide specific evidence in 2-3 sentences>"
                }},
                {{
                    "criterion": "Compelling Scientific Question",
                    "score": <1-5>,
                    "justification": "<Does it address novel/significant questions? Provide specific evidence in 2-3 sentences>"
                }},
                {{
                    "criterion": "Open Science",
                    "score": <1-5>,
                    "justification": "<Commitment to making findings publicly available? Provide specific evidence in 2-3 sentences>"
                }},
                {{
                    "criterion": "Training Component",
                    "score": <1-5>,
                    "justification": "<Quality of training opportunities for graduate students/postdocs? Provide specific evidence in 2-3 sentences>"
                }},
                {{
                    "criterion": "Need for Support",
                    "score": <1-5>,
                    "justification": "<Clear justification for why NCEMS support is needed? Provide specific evidence in 2-3 sentences>"
                }}
                ]
            }}
            }}

            Provide ONLY the JSON output above with no additional text before or after.""",
                        parameters=['research_call', 'proposal_id', 'proposal_title',  'proposal_abstract', 
                                'proposal_full', 'role_description']
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
    
    def format_prompt(self, template_name: str, data: Dict[str, Any], role: str = None) -> str:
        """Format a prompt using a specific template, data, and role"""
        template = self.get_template(template_name)
        
        # Prepare base data for formatting
        format_data = {
            'research_call': data.get('research_call', 'N/A'),
            'title': data.get('title', 'N/A'),
            'abstract': data.get('abstract', 'N/A')
        }
        
        # Only add role_description if the template requires it
        if template.parameters and 'role_description' in template.parameters:
            # Template requires role_description - use default if not specified
            if role is None:
                role = 'single_scientist'
            if role not in self.role_configs:
                raise ValueError(f"Role '{role}' not found. Available: {list(self.role_configs.keys())}")
            format_data['role_description'] = self.role_configs[role]['role_description']
        # Note: If template doesn't require role_description (like generate_ideas_no_role),
        # this entire block is skipped and no role is used
        
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