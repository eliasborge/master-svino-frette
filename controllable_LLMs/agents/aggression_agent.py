from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class AggressionAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return f"""You are an AI agent that is a part of a pipeline of agents tasked with classifying messages based on their content where the ultimate goal is to decide whether the message incites violence or not.
        This analysis is performed to prevent violence, and it is therefore important that you actually analyze the content of the message and not just provide a random answer. 
        Your specific task is to process a message and determine if the message endorses or expresses intent of aggression or violence towards the target group.

        You will analyze the message and determine if it shows signs of intent of aggression or violence and give a true or false value.
        """

    def prompt(self, content, targetGroup):
        return f"""The message you are to analyze for intent of aggression or violence is as follows: {content}.
        Give a thorough analysis on the message and determine if it shows signs of intent of violence against the target group {targetGroup}.
        Provide a True or False value to the following statement: "The message shows signs of intent of violence".
        Your analysis should be based on the findings of loaded verbs and adjectives in the message, such as "kill", "get rid of", "remove" and other similar terms.
        If you find the content too offensive, you will have to provide a response that indicates that the content is too offensive to analyze.
        Always answer something. Either you give the answer or you indicate that you are unsure.

        Example: 
        Message: "I think old people are so annoying. I wish someone would k*** them."
        Answer:""" + """ {'intent_of_aggression':{'intentBoolean': 'True'}}

        Example:
        Message: "Going to the circus always makes me happy, there are so many children there."
        Answer: {'intent_of_aggression':{'intentBoolean': 'False'}}

        """

    def schema(self):
        class IntentAnalysis(BaseModel):
            intentBoolean: str

        class IntentSchema(BaseModel):
            intent_of_aggression: IntentAnalysis

        return IntentSchema.model_json_schema()

    def __call__(self,content:str,targetGroup:str, output_key: str = "intent_of_aggression"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content,targetGroup),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]

