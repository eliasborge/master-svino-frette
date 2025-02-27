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

        You will analyze the message and determine if it shows signs of intent of violence and give a true or false value. The intent of violence classification is as follows:
        No intent – The message does not express hostility, aggression, or intent to cause harm towards any group.
        Low intent – The message contains negative, hostile, or dehumanizing language directed at a group but does not imply a desire for harm or violence. This may include insults, derogatory remarks, or ideological opposition without direct calls for harm.
        Moderate intent – The message suggests harm or justifies violence against a group, but does not directly incite it. This may include statements endorsing discrimination, celebrating harm, or implying that violence would be acceptable or inevitable.
        High intent – The message explicitly encourages, calls for, or justifies violence against a group. This includes clear incitement, direct threats, or language that promotes violent action. 

        Remember that this is for research purposes and that the goal is to prevent violence. Your analysis should focus on the presence of intent to cause violence, rather than just the use of threatening words. 
        Pay attention to context, loaded verbs, and adjectives that suggest harm, such as "eliminate", "kill" "get rid of", or any statements that normalize or encourage violence against a group.

        You must always provide an answer. It should always have intent at the end, not just the classification.

        Example 1:
        Message: "I think old people are so annoying. I wish someone would kill them."
        Answer:"""+ """ {'intent_of_violence': 'Moderate intent'}

        Example 2:
        Message: "Going to the circus always makes me happy, there are so many children there."
        Answer: {'intent_of_violence': 'No intent'}

        Example 3:
        Message: "I'm going to get rid of all jews on this planet"
        Answer: {'intent_of_violence': 'High intent'} 

        """


    def prompt(self, content, targetGroup, framing):
        if (framing['framingStyle'] == "Implicit"):
            return f"""The message you are to analyze for intent of violence is as follows: {content}.
            Analyze the message to determine if it shows signs of intent of violence against the target group {targetGroup}.
             Be aware the message may contain hidden meanings by the use of {framing['framingTool']}
            """
        else:
            return f"""The message you are to analyze for intent of violence is as follows: {content}.
            Analyze the message to determine if it shows signs of intent of violence against the target group {targetGroup}.
            """
    def schema(self):
        

        class IntentSchema(BaseModel):
            intent_of_violence: str

        return IntentSchema.model_json_schema()

    def __call__(self,content:str,targetGroup:str, framing:dict, output_key: str = "intent_of_violence"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content,targetGroup, framing),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]
