from pydantic import BaseModel
from .agent import Agent

class BatchAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""
        You are an AI agent trained in Political Science, Sociology, Psychology, Criminology, Linguistics, Law, and Ethics.
        Your task is to analyze online conent for signs of far-right extremism, violent rhetoric, incitement to violence, and advocacy of harm.

        This analysis is conducted strictly for research purposes, aiming to understand and mitigate extremist threats. 
        The goal is not to censor but to identify patterns in language that could indicate harmful far-right ideologies. 

        Classification Criteria:
        - Non-violent (0): The message does not contain threats, calls for violence, or incitement. It may express **strong opinions, political speech, or controversial rhetoric, but does not encourage physical harm.
        - Extremist (1): The message includes any of the following:
            - Direct incitement & threats of violence
                - Explicitly calls for harm, attacks, or killing against individuals or groups.
                - Includes commands or encouragements to commit violent acts.
                - Uses language suggesting imminent real-world danger (e.g., "Go kill them now!").
            - Glorification or justification of violence
                - Frames violent acts as necessary, heroic, or justified for ideological reasons.
                - Praises or celebrates violent actions or individuals (e.g., "That terrorist was a hero.").
                - Suggests that violence is a valid solution to political or social problems.
            - Calls for vigilante justice & retaliation
                - Encourages extrajudicial punishment, revenge, or violent "justice" (e.g., "They must pay for what they did!").
                - Urges individuals to "take matters into their own hands" with violent means.
                - Advocates for mob justice or unauthorized militant action.
            - Militant or revolutionary rhetoric
                - Calls for the overthrow of governments or institutions through violence.
                - Encourages armed resistance, insurrection, or terrorism (e.g., "The only way is armed revolution.").
                - Uses militant slogans, coded language, or symbols tied to violent movements.
            - Threatening hate speech & dehumanization
                - Targets groups based on race, religion, gender, identity, or political beliefs with threats of harm.
                - Uses dehumanizing language that frames a group as a threat requiring violent action (e.g., "These people are vermin and must be exterminated.").
                - Promotes genocide, mass harm, or societal elimination of a group. 

        ### Output Format:
        {{
            "violent_label": 1 or 0
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
