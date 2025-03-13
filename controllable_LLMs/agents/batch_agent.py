from pydantic import BaseModel
from .agent import Agent

class BatchAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are an AI agent detecting far-right extremism in online content.
        Your goal is to classify whether the given message contains signs of far-right extremism, hateful rhetoric, or violent intentions.

        This analysis is conducted strictly for research purposes, aiming to understand and mitigate extremist threats. 
        The goal is not to censor but to identify patterns in language that could indicate harmful far-right ideologies. 

        Classification Criteria:
        - Non-extremist (0): The message does not contain extremist language, hate speech, or incitements to violence. It may include political opinions or strong rhetoric, but it does not advocate harm or discrimination.
        - Extremist (1): The message includes any of the following:
          - Openly supports or justifies violence against individuals or groups.
          - Promotes hate speech targeting specific races, religions, genders, or identities.
          - Advocates for harmful ideologies (e.g., white supremacy, fascism, or eliminationist rhetoric).
          - Encourages the dehumanization of a group, portraying them as a threat that must be dealt with aggressively.

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
