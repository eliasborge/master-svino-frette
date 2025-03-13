from pydantic import BaseModel
from .agent import Agent

class BatchAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are an AI agent detecting far-right extremism in online content.
        Your task is to classify whether a given text contains extremist language, ideology, or hate speech.

        ### Classification Rules:
        - If the text contains extremist views, hate speech, calls for violence, or conspiracy theories, classify as `1`.
        - Otherwise, classify as `0`.

        ### Output Format:
        {{
            "extremism_label": 1 or 0
        }}
        """

    def prompt(self, content):
        return f"Analyze this text and classify it as extremist (1) or non-extremist (0): {content}"

    def schema(self):
        class ExtremismResult(BaseModel):
            extremism_label: int

        class ExtremismSchema(BaseModel):
            result: ExtremismResult

        return ExtremismSchema.model_json_schema()

    def __call__(self, content, output_key: str = "result"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]['extremism_label']
