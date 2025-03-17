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

        Labeling: 
        Answer "No Call to Action" if it is a No call-to-action
        Answer "Low Call to Action" if it is a Low call-to-action
        Answer "Moderate Call to Action" if it is a Moderate call-to-action
        Answer "High Call to Action" if it is a High call-to-action

        """

    def prompt(self, content, targetGroup, framing, context,mode):
        if mode == "no-context":
            if framing['framingStyle'] == "Implicit":
                return f"""
                The message you are to analyze for call-to-action rhetoric is as follows: {content}.
                Analyze whether the message urges others to take action against the target group {targetGroup}.
                Be aware that the message may contain indirect or subtle calls to action using {framing['framingTool']}."""
            else:
                return f"""
                The message you are to analyze for call-to-action rhetoric is as follows: {content}.
                Analyze whether the message urges others to take action against the target group {targetGroup}."""

        elif(mode=="context"):
            if framing['framingStyle'] == "Implicit":
                return f"""
                You have been given a message that is a part of a broader conversation. This conversation has been analyzed by a context agent
                to provide you with insights into how relevant the surrounding messages are to the classification of this message.
                The context is as follows: {context}

                The message you are to analyze for call-to-action rhetoric is as follows: {content}.
                Analyze whether the message urges others to take action against the target group {targetGroup}.
                Be aware that the message may contain indirect or subtle calls to action using {framing['framingTool']}."""
            else:
                return f"""
                You have been given a message that is a part of a broader conversation. This conversation has been analyzed by a context agent
                to provide you with insights into how relevant the surrounding messages are to the classification of this message.
                The context is as follows: {context}

                The message you are to analyze for call-to-action rhetoric is as follows: {content}.
                Analyze whether the message urges others to take action against the target group {targetGroup}."""
        elif(mode=="neighbor"):
            if framing['framingStyle'] == "Implicit":
                return f"""
                You have been given a message that is part of a series of neighboring messages. These neighboring messages provide additional context that may help in the classification of the message.
                The thread as follows: {context}

                The message you are to analyze for call-to-action rhetoric is as follows: {content}.
                Analyze whether the message urges others to take action against the target group {targetGroup}.
                Be aware that the message may contain indirect or subtle calls to action using {framing['framingTool']}."""
            else:
                return f"""
                You have been given a message that is part of a series of neighboring messages. These neighboring messages provide additional context that may help in the classification of the message.
                The thread as follows: {context}

                The message you are to analyze for call-to-action rhetoric is as follows: {content}.
                Analyze whether the message urges others to take action against the target group {targetGroup}."""

    def schema(self):
        class CallToActionSchema(BaseModel):
            call_to_action: int
        return CallToActionSchema.model_json_schema()

    def __call__(self, content: str, targetGroup: str, framing: dict,context,mode:str, output_key: str = "call_to_action"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(content, targetGroup, framing, context,mode),
            schema=self.schema(),
            model=self.model
        )
        if output:
            return output[output_key]
