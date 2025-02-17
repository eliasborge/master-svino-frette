from pydantic import BaseModel, Field, conlist
from typing import List, Dict, Any, Literal
from agent import Agent
from api_extraction import model

def filter_labels(filtered_datasets: List[Dict[str, Any]]):
    filtered_labels = set()
    for dataset in filtered_datasets:
        for label in dataset["labels"]:
            filtered_labels.add(label["label"].lower())

    return list(filtered_labels)

class LabelerAgent(Agent):
    def __init__(
        self, model, topic: str, num_labels: int
    ):
        super().__init__(model)
        self.topic = topic
        self.num_labels = num_labels

    def system(self):
        return f"You are an assistant that labels datasets based on the topic of {self.topic}."

    def prompt(self):
        return f"Create a list of {self.num_labels} label categories for a task related to {self.topic}. Determine the name, a description, and list of possible values. There must be exactly {self.num_labels} values. Output in JSON."

    def schema(self):
        class DatasetLabel(BaseModel):
            name: str
            description: str
            possible_values: List[str]

        class LabelerSchema(BaseModel):
            # labels: List[DatasetLabel]
            labels: conlist(DatasetLabel, min_length=self.num_labels, max_length=self.num_labels)

        return LabelerSchema.model_json_schema()

    def __call__(self, output_key: str = "labels"):
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

labels = LabelerAgent(model=model, topic="sentiment analysis", num_labels=4)()
for label in labels:
    print(label)
    print()