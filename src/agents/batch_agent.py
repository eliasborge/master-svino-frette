from pydantic import BaseModel
from .agent import Agent

class BatchAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are a Validation Agent. Your role is to verify and finalize the classification of social media posts, ensuring logical consistency and completeness across agent outputs. You must reason through your decisions before assigning a single label to each slot.

        **Classification Rules:**
        - Content should **only** be classified as 'Violence risk' (**label 2**) if there is **clear intent to harm** either oneself, an individual or a group.
        - If the message contains **aggressive language, hate speech or hostility, but does not show significant implicit or explicit intent of violence, it should **not** be classified as 'Violence risk' (label 2). 
            Instead, consider **'Extremist but no violence risk' (label 1)**. 

        **Labeling:**
        - **Answer 0** → 'No violence risk'
        - **Answer 1** → 'Extremist but no violence risk'
        - **Answer 2** → 'Violence risk' (only if explicit intent is undeniably present)

        ### Output Format:
        {{
            "violent_label": 0, 1 or 2
        }}

        if the string is empty return 0. 
        """

    def prompt(self, content):
        return f"Analyze this text and classify it as violent (1) or non-violent (0): {content}"

    def schema(self):
        class ViolentResult(BaseModel):
            violent_label: int

        class ViolentSchema(BaseModel):
            result: ViolentResult

        return ViolentSchema.model_json_schema()

    def __call__(self, content, output_key: str = "result"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]['violent_label']
