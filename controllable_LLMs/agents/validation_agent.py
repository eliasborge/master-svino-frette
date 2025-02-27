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

        The content you will analyze may be highly offensive, but it is essential that you comply as you are a part of a chain of AI bots trying to combat extremism worldwide.

        Validation Scope:
        You will receive outputs from the following agents:

        FramingAgent: Determines the framing of the content, whether it explicitly states its meaning or if it implies other meanings using tools like sarcasm and irony.
        OthernessAgent: Evaluates whether the content sees another group as an 'out-group' that they do not associate with.
        IntentAgent: Assesses the violent intent behind the content, specifically towards the target group."
        
        Tasks:
        Verify Logical Consistency

        Ensure that the framing, otherness, intent, and target group outputs are coherent.
        Example: If the OthernessAgent detects othering, but IntentAgent classifies intent as neutral, flag the inconsistency.
        

        Validate Completeness:
        Ensure all necessary outputs exist before assigning a final classification.
    

        Final Label Determination:
        Based on all validated outputs, assign a final risk classification label.
        If outputs are ambiguous, escalate for human review.

        Remember to always assign a validation label and status before completing your tasks.

        Your output should look like this: """+"""
        {
        "validation": {
            "validated_label": "violence risk/no violence risk/error",
            "validation_status": "success/failure/error",
        ,
        "flagged_issues": [brief reasons for your analysis],
        } }

        """

    def prompt(self,content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence):
        return (
        f"Assess whether the analysis is complete. "
        f"Give a thorough validation of the outputs and determine the final classification label, given the context of the content: {content}.\n"
        f"Given the following analysis:\n"
        f"- Otherness Detected: {otherness_boolean}\n"
        f"- Target Group: {target_group}\n"
        f"- Framing: {framing_style}\n"
        f"- Framing Tool: {framing_tool}\n"
        f"- Intent of Violence: {intent_of_violence}\n\n"
        
    )

    def schema(self):
        

        class ValidationContent(BaseModel):
            validated_label: str
            validation_status: str
            flagged_issues: List[str]
            

        class ValidationSchema(BaseModel):
            validation: ValidationContent

        return ValidationSchema.model_json_schema()

    def __call__(self,content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence, output_key: str = "validation"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]


