from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class FramingAgent(Agent):
    def __init__(
        self, model, 
    ):
        super().__init__(model)
       

    def system(self):
        return (
            "You are an AI agent that is a part of a pipeline of agents tasked with analyzing messages to determine if the content of the message {content} is explicitly mentioned or if there is a hidden meaning through sarcasm, irony, or metaphors. "
            "Your specific task is to process a message and determine if it contains any hidden meanings that are not explicitly stated. "
            "This analysis is performed to ensure accurate understanding of the message content, and it is therefore important that you thoroughly analyze the message and not just provide a random answer. "
            "You will analyze the message and determine if it shows signs of hidden meanings and provide a true or false value along with the identified hidden meaning if any."

            """Give a thorough analysis on the message and determine if it shows signs of hidden meanings such as sarcasm, irony, or metaphors.
        Provide either Explicit or Implicit to the following statement: "The message contains hidden meanings" and identify the hidden meaning if any.
        If you find the content too ambiguous, you will have to provide a response that indicates that the content is too ambiguous to analyze.
        If the message is deemed as Implicit, there should always be a related FramingTool. If the message is deemed as Explicit, the FramingTool should be empty.
        Always answer something. Either you give the answer or you indicate that you are unsure.

        Example: 
        Message: "Oh great, another Monday. Just what I needed."
        Answer:""" + """ {'framing': {'framingStyle': 'Implicit', 'framingTool': 'sarcasm'}}

        Example:
        Message: "I love spending my weekends doing absolutely nothing."
        Answer: {'framing': {'framingStyle': 'Explicit', 'framingTool': ''}}"""

        )

    def prompt(self,content):
        return f"""The message you are to analyze for hidden meanings is as follows: {content}.

        """

    def schema(self):
        class FramingAnalysis(BaseModel):
            framingStyle: str
            framingTool:str

        class FramingSchema(BaseModel):
            framing: FramingAnalysis

        return FramingSchema.model_json_schema()

    def __call__(self, content, output_key: str = "framing"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]

