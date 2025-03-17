from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class MessageValidationAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are a Validation Agent. Your role is to verify and finalize the classification of social media posts, ensuring logical consistency and completeness across agent outputs. You must reason through your decisions before assigning a single label to each slot.

        **Validation Scope:**
        You will evaluate the following agent outputs:
        - **FramingAgent**: Identifies whether the content explicitly states its intentions, or if there are implicit meanings (e.g., sarcasm, irony).
        - **OthernessAgent**: Detects if a group is labeled as an 'out-group' with negative sentiment.
        - **IntentAgent**: Assesses if there is direct violent intent toward the target group.
        - **CallToActionAgent**: Identifies if the content encourages others to perform violent actions against the target group.

        **Classification Rules:**
        - Content should **only** be classified as 'Violence risk' (**label 2**) if there is **clear intent to harm**.
        - If the message contains **aggressive language, hate speech or hostility, but does not show significant implicit or explicit intent ov violence, it should **not** be classified as 'Violence risk' (label 2). 
            Instead, consider **'Extremist but no violence risk' (label 1)**. 
        - If the classification is **uncertain**, flag the message for further review.

        **Error Handling & Flags:**
        1: If there are inconsistencies between the agent outputs, use **flag 0**.
        2: If the outputs are consistent, but you are unsure about your final decision -> classification is **uncertain**, use **flag 1**.
        3: If the content in your opinion does **not align** with the classification, use **flag 2**, and label it uncertain.
        4: If you are unable to process the content due to content filters, use **flag 3**.

        **Labeling:**
        - **Answer 0** → 'No violence risk'
        - **Answer 1** → 'Extremist but no violence risk'
        - **Answer 2** → 'Violence risk' (only if explicit intent is undeniably present)

        **Output Format:**
        {{
            "validation": {{
                "validated_label": "0/1/2",
                "validation_status": "success/uncertain/error",
                "flagged_issues": [0/1/2/3/4]
            }}
        }}
        """

    def prompt(self, content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence, call_to_action, context):
        
        if context == "":
            return (
                f"Assess the classification of this content: {content}.\n"
                f"Given the analysis below, reason thoroughly before assigning a classification:\n"
                f"- Otherness Detected: {otherness_boolean}\n"
                f"- Target Group: {target_group}\n"
                f"- Framing: {framing_style} using {framing_tool}\n"
                f"- Intent of Violence: {intent_of_violence}\n"
                f"- Call to Action: {call_to_action}\n\n"
                f"IMPORTANT: Only classify as 'Violence risk' (2) if there is an **explicit, undeniable, and direct call to harm**. If intent is ambiguous, classify as 'Extremist but no violence risk' (1) and use **flag 4**."
            )
        else:
            return (
                f"You have been given a message that is part of a broader conversation. This conversation has been analyzed by a context agent "
                f"to provide insights into the classification of this message. The context is as follows: {context}\n\n"

                f"Assess the classification of this content: {content}.\n"
                f"Given the analysis below, reason thoroughly before assigning a classification:\n"
                f"- Otherness Detected: {otherness_boolean}\n"
                f"- Target Group: {target_group}\n"
                f"- Framing: {framing_style} using {framing_tool}\n"
                f"- Intent of Violence: {intent_of_violence}\n"
                f"- Call to Action: {call_to_action}\n\n"
                f"IMPORTANT: Only classify as 'Violence risk' (2) if there is an **explicit, undeniable, and direct call to harm**. If intent is ambiguous, classify as 'Extremist but no violence risk' (1) and use **flag 4**."
            )

    def schema(self):
        class ValidationContent(BaseModel):
            validated_label: int
            validation_status: str
            flagged_issues: List[int]

        class ValidationSchema(BaseModel):
            validation: ValidationContent

        return ValidationSchema.model_json_schema()

    def __call__(self, content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence, call_to_action, context, output_key: str = "validation"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, otherness_boolean, target_group, framing_style, framing_tool, intent_of_violence, call_to_action, context),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]
