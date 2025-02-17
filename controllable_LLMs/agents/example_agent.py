from pydantic import BaseModel, conlist
from typing import List
from controllable_LLMs.agents.agent import Agent


class LabelerAgent(Agent):
    def __init__(
        self, model, topic: str, num_labels: int
    ):
        super().__init__(model)
        self.topic = topic
        self.num_labels = num_labels

    def system(self):
        return f"SYSTEM MESSAGE"

    def prompt(self):
        return f"PROMPT"

    def schema(self): # EXAMPLE WITH LABEL GENERATION
        class DatasetLabel(BaseModel):
            name: str
            description: str
            possible_values: List[str]

        class LabelerSchema(BaseModel):
            # labels: List[DatasetLabel]
            labels: conlist(DatasetLabel, min_length=self.num_labels, max_length=self.num_labels)

        return LabelerSchema.model_json_schema()

    def __call__(self, output_key: str = "OUTPUT_VARIABLE_NAME"):
        output = self.generate(
            system_prompt=self.system(),
            prompt=self.prompt(),
            schema=self.schema(),
            model=self.model,
            num_ctx=200,
            temperature=0.0,
        )
        if output:
            return output[output_key]

