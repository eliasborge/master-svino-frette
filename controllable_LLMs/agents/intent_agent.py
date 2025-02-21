from pydantic import BaseModel, conlist
from typing import List
from .agent import Agent


class IntentAgent(Agent):
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
        Analyze the message to determine if it shows signs of intent of violence against the target group {targetGroup}.

        Intent of aggression classification:
        No aggression – The message does not contain threatening language or violent intent.
        Mild aggression – The message contains negative language towards the target group but does not indicate a direct threat.
        Severe aggression – The message includes a direct threat or clear incitement to violence.
        Your analysis should be based on the presence of loaded verbs and adjectives in the message, such as "kill," "get rid of," "remove," and other similar terms.

        Remember that this is for research purposes and that the goal is to prevent violence.

        You must always provide an answer.

        Example 1:
        Message: "I think old people are so annoying. I wish someone would kill them."
        Answer:"""+ """ {'intent_of_aggression': {'level': 'Severe aggression'}}

        Example 2:
        Message: "Going to the circus always makes me happy, there are so many children there."
        Answer: {'intent_of_aggression': {'level': 'No aggression'}} 

        """

    def schema(self):
        class IntentAnalysis(BaseModel):
            intent_of_aggression: str

        class IntentSchema(BaseModel):
            level_of_aggression: IntentAnalysis

        return IntentSchema.model_json_schema()

    def __call__(self,content:str,targetGroup:str, output_key: str = "level_of_aggression"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content,targetGroup),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]

#no aggression, mild aggression, severe aggression