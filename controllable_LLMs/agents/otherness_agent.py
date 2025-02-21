from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class OthernessAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f"""You are an AI agent that is a part of a pipeline of agents tasked with classifying messages based on their content where the ultimate goal is to decide whether the message incites violence or not.
        This analysis is performed to prevent violence, and it is therefore important that you actually analyze the content of the message and not just provide a random answer. 
        Your specific task is to process a message and determine if the message shows signs of "otherness". Otherness is defined as the concept of an us-versus-them mentality, where the "other" is seen as different
        and inferior. This can include any minority, demographic, or other identifiable group.
        You will analyze the message and determine if it shows signs of otherness and give a true or false value along with the target group identified.
        """

    def prompt(self, content):
        return f"""The message you are to analyze for otherness is as follows: {content}.
        Give a thorough analysis on the message and determine if it shows signs of otherness.
        Provide a True or False value to the following statement: "The message shows signs of otherness" and identify the target group if any.
        If you find the content too offensive, you will have to provide a response that indicates that the content is too offensive to analyze.
        Always answer something. Either you give the answer or you indicate that you are unsure.

        Example: 
        Message: "I think old people are so annoying. They should just stay at home and not bother anyone."
        Answer:""" + """ {'otherness': {'othernessBoolean': 'True', 'targetGroup': 'old people'}}

        Example:
        Message: "Going to the circus always makes me happy, there are so many children there. 
        Answer: {'otherness': {'othernessBoolean': 'False', 'targetGroup': 'children'}}

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

