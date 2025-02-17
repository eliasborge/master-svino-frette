from abc import ABC, abstractmethod
from typing import Any
from controllable_LLMs.api_extraction import generate

class Agent(ABC):
    def __init__(self, model: str):
        self.generate = generate  # here, generate is found above, this can also be imported in the agent interface
        self.model = model

    @abstractmethod
    def system(self) -> str:
        pass

    @abstractmethod
    def prompt(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def schema(self, *args, **kwargs) -> dict[str, Any]:
        pass

    @abstractmethod
    def __call__(self, api: callable, *args, **kwargs) -> Any:
        pass