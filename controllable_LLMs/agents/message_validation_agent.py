from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class MessageValidationAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are a Validation Agent. Your role is to verify and finalize the classification of social media posts, ensuring logical consistency and completeness across agent outputs. You must reason through your decisions before assigning a single label to each slot.

        Validation Scope:
        You will evaluate the following agent outputs:
        - FramingAgent: Identifies whether the content explicitly states its intentions, or if there are implicit meanings through (e.g., sarcasm, irony).
        - OthernessAgent: Detects if a group is labeled as an 'out-group' with negative sentiment.
        - IntentAgent: Assesses if there is direct violent intent toward the target group.
        - CallToActionAgent: Identifies if the content encourages others to perform violent actions against the target group.

        Tasks:
        1. Ensure consistency: Verify that all agent outputs align logically (e.g., if 'otherness' is "False", it is likely that 'intent' should not be "High").
        2. Ensure completeness: Confirm that all necessary outputs are present before determining the final classification.
        3. Reason before assigning labels: Review all inputs and reason thoroughly before assigning a classification label.
        4. when assigning a label, provide a single label per slot.

        Error handling:
        1: If there are inconsistensies between the agent outputs, use flag 0.
        2: If the outputs are consistent, but the classification is uncertain, use flag 1.
        3: If the content in your opinion does not align with the classification, use flag 2, and label it uncertain.
        4: If you are unable to process the content due to content filters, use flag 3.

        Labeling:
        Answer 0 if the content is classified as 'No violence risk'
        Answer 1 if the content is classified as 'Extremist but no violence risk'
        Answer 2 if the content is classified as 'Violence risk'




        Output format:
        {{
            "validation": {{
                "validated_label": "0/1/2",
                "validation_status": "success/uncertain/error",
                "flagged_issues": [1/2/3/4]
            }}
        }}
        """

    def prompt(self, content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence, call_to_action, context):
        return (
            f"You have been given a message that is a part of a broader conversation. This conversation has been analyzed by a context agent"
            f"to provide you with insights into how relevant the surrounding messages are to the classification of this message."
            f"The context is as follows: {context}"

            f"Assess the classification of this content: {content}.\n"
            f"Given the analysis below, reason thoroughly and provide a single label per slot:\n"
            f"- Otherness Detected: {otherness_boolean}\n"
            f"- Target Group: {target_group}\n"
            f"- Framing: {framing_style} using {framing_tool}\n"
            f"- Intent of Violence: {intent_of_violence}\n"
            f"- Call to Action: {call_to_action}\n\n"
        )

    def schema(self):
        class ValidationContent(BaseModel):
            validated_label: int
            validation_status: str
            flagged_issues: List[str]

        class ValidationSchema(BaseModel):
            validation: ValidationContent
            validation_status: str
            flagged_issues: List[str]
            

        class ValidationSchema(BaseModel):
            validation: ValidationContent

        return ValidationSchema.model_json_schema()

    def __call__(self,content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence, call_to_action,context, output_key: str = "validation"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence, call_to_action, context),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]


