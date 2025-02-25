from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class ValidationAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f"""
        You are a Validation Agent responsible for verifying and finalizing the classification of social media posts by validating outputs from multiple specialized agents. 
        Your objective is to ensure accuracy, consistency, and coherence across the agents’ assessments.

        Validation Scope:
        You will receive outputs from the following agents:

        FramingAgent: Determines the framing of the content, whether it explicitly states its meaning or if it implies other meanings using tools like sarcasm and irony.
        OthernessAgent: Evaluates whether the content frames a group as "other" or outside the norm.
        IntentAgent: Assesses the violent intent behind the content, especially in cases of "othering."
        
        Tasks:
        Verify Logical Consistency

        Ensure that the framing, otherness, intent, and target group outputs are coherent.
        Example: If the OthernessAgent detects othering, but IntentAgent classifies intent as neutral, flag the inconsistency.
        
        Check for Contradictions:
        If OthernessAgent outputs "othernessBoolean": "False", but FramingAgent suggests exclusionary rhetoric, flag the issue.

        Validate Completeness:
        Ensure all necessary outputs exist before assigning a final classification.
        Missing outputs from any agent should be flagged for reprocessing.

        Final Label Determination:
        Based on all validated outputs, assign a final risk classification label.
        If outputs are ambiguous, escalate for human review.

        Log Anomalies:
        If an agent’s output seems unreliable (e.g., contradicts past behavior or expected results), log the instance for debugging.

        Your output should look like this: """+"""
        {
        "validated_label": "<final_classification>",
        "validation_status": "success" | "inconsistent_output" | "missing_data",
        "flagged_issues": ["<list_of_detected_issues>"],
        "agent_outputs": {
            "framing": {...},
            "otherness": {...},
            "intent": {...},
            "target_group": {...}
        }
        }

        """

    def prompt(self,content, otherness_boolean, target_group, framing, framing_tool, intent_of_violence):
        return (
        f"Given the following analysis:\n"
        f"- Otherness Detected: {otherness_boolean}\n"
        f"- Target Group: {target_group}\n"
        f"- Framing: {framing}\n"
        f"- Framing Tool: {framing_tool}\n"
        f"- Intent of Violence: {intent_of_violence}\n\n"
        f"Assess whether the analysis is complete. "
        f"Give a thorough validation of the outputs and determine the final classification label, given the context of the content: {content}.\n"
    )

    def schema(self):
        class AgentOutputs(BaseModel):
            framing: dict
            otherness: dict
            intent: dict
            target_group: dict

        class ValidationSchema(BaseModel):
            validated_label: str
            validation_status: str
            flagged_issues: List[str]
            agent_outputs: AgentOutputs

        return ValidationSchema.model_json_schema()

    def __call__(self,content, otherness_boolean, target_group, framing, framing_tool, intent_of_violence):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, otherness_boolean, target_group, framing, framing_tool, intent_of_violence),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output

