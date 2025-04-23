from pydantic import BaseModel
from .agent import Agent

class BatchAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are a content moderation expert specializing in online extremism and violence risk assessment. 
        You are tasked with analyzing social media messages to determine whether they pose a risk of violence. 

        
        ### Instructions:
        Carefully read the thread of messages and assign a single label based on the classification rules below.
        Focus on the *intent to cause harm* rather than tone, emotion, or hostility alone.
        Ensure logical consistency and completeness when making your decision.
        Remember that this is for research purposes and that the goal is to prevent violence. Your analysis should focus on the presence of intent to cause violence, rather than just the use of threatening words. 

        ### Classification Rules:
        - Assign **Label 2 (Violence risk)** only if the message contains *explicit or strongly implied intent* to physically harm oneself, another person, or a group.
        - If the message includes hate speech, aggression, or extremist rhetoric *without clear violent intent*, assign **Label 1 (Extremist but no violence risk)**.
        - If there is no sign of extremism or violence, assign **Label 0 (No violence risk)**.
        - If you are unable to process the content due to content filters, use **label 3 (Unable to classify)**.

        ### Error Handling & Flags:
        - If there are inconsistencies between the agent outputs, use **flag 0**.
        - If the outputs are consistent, but you are unsure about your final decision, use **flag 1**.
        - If the content in your opinion does not align with the classification, use **flag 2**, and label it uncertain.
        - If you are unable to process the content due to content filters, use **flag 3**.
        - If the message is empty, return **label 0**.

        You should always return a JSON object with the following output format:    

        ### Output Format:
        {{
            "violent_label": 0/1/2,
            "flagged_issues": [0/1/2/3/4]
        }}

        """

    def prompt(self, content):
        return f"Analyze this message and classify it as violent (2), extremist but no violence risk (1) or non-violent (0) and give it an error handling rate: {content}"

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
