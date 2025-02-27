from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class OthernessAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f""" You are an AI agent that is part of a pipeline of agents tasked with classifying messages based on their content. 
        The ultimate goal is to assess whether a message incites violence, and your specific role is to determine if the message exhibits "otherness." Otherness is the concept of an "us-versus-them" mentality, where a group is framed as different, separate, or inferior.  
        This framing can involve minorities, demographics, social groups, political affiliations, or any identifiable community.  

        Important distinctions to consider:
        - Otherness is only True if there is negative framing of another group.  
        - If a message is neutral or positive towards a group without attacking another, otherness should be False.  
        - If the message contains positive sentiment about an in-group (e.g., "I love my own group"), otherness should be False unless it simultaneously degrades another group.

        Your task is to analyze the message and determine if it exhibits signs of otherness. If otherness is present, you must identify the target group. The response must always include a True or False value for otherness and the target group if otherness is True.
        You cannot return an empty target group if otherness is True, as this is an inconsistency.
        
        Output format:

        {{
            "otherness": {{
                "othernessBoolean": "True/False",
                "targetGroup": "group_name"
            }}
        {{

        """

    def prompt(self, content):
        return f"""The message you are to analyze for otherness is as follows: {content}.
        Give a thorough analysis on the message and determine if it shows signs of otherness.

        Provide a True or False value to the following statement: "The message shows signs of otherness" and identify the target group if any. Remember that otherness is true only if another group is framed negatively.
        If a group is mentioned neutrally or positively, otherness should be False. If a message expresses positive sentiment toward an in-group without attacking another group, otherness should be False.
        If you find the content too offensive, you will have to provide a response that indicates that the content is too offensive to analyze.
        Always answer something. If you find otherness to be true, there has to be an associated target group. i.e. you cannot answer otherness = true and target group = "", as this is an inconsistency

        Output format:

        {{
            "otherness": {{
                "othernessBoolean": "True/False",
                "targetGroup": "group_name"
            }}
        {{
        """

    def schema(self):
        class OthernessAnalysis(BaseModel):
            othernessBoolean: str
            targetGroup: str

        class OthernessSchema(BaseModel):
            otherness: OthernessAnalysis

        return OthernessSchema.model_json_schema()

    def __call__(self,content:str, output_key: str = "otherness"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]

