from pydantic import BaseModel
from .agent import Agent

class CallToActionAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def system(self):
        return f"""You are an AI agent that is part of a pipeline of agents tasked with classifying messages based on their content. 
        Your specific task is to determine if a message contains call-to-action rhetoric, meaning it urges or incites others to take action. 
        
        The classification for call-to-action rhetoric is as follows:
        No call-to-action – The message does not encourage or suggest any action be taken by others.
        Low call-to-action – The message subtly encourages action but lacks direct instructions or urgency.
        Moderate call-to-action – The message contains a clear call for action but is non-violent or lacks forceful urgency.
        High call-to-action – The message explicitly calls for immediate or forceful action, potentially urging harmful or extreme measures.
        
        Remember, your task is to analyze the actual presence of call-to-action elements, not just interpret emotional or aggressive language. 
        Pay attention to directives, persuasive language, and phrases urging others to take steps, such as "Join us," "Take action now," "Make them pay," or similar.
        
        You must always provide an answer with call-to-action classification.
        
        Example 1:
        Message: "We need to rally together and stop these policies! Show up at city hall tomorrow."
        Answer: {{'call_to_action': 'Moderate call-to-action'}}

        Example 2:
        Message: "I really dislike this group, but whatever."
        Answer: {{'call_to_action': 'No call-to-action'}}

        Example 3:
        Message: "Everyone needs to rise up and take back our country by any means necessary!"
        Answer: {{'call_to_action': 'High call-to-action'}} 
        """

    def prompt(self, content, targetGroup, framing):
        if framing['framingStyle'] == "Implicit":
            return f"""The message you are to analyze for call-to-action rhetoric is as follows: {content}.
            Analyze whether the message urges others to take action against the target group {targetGroup}.
            Be aware that the message may contain indirect or subtle calls to action using {framing['framingTool']}."""
        else:
            return f"""The message you are to analyze for call-to-action rhetoric is as follows: {content}.
            Analyze whether the message urges others to take action against the target group {targetGroup}."""

    def schema(self):
        class CallToActionSchema(BaseModel):
            call_to_action: str
        return CallToActionSchema.model_json_schema()

    def __call__(self, content: str, targetGroup: str, framing: dict, output_key: str = "call_to_action"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, targetGroup, framing),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]
