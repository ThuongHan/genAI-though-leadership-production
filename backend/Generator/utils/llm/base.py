from abc import ABC, abstractmethod

class BaseLLM(ABC):

    @abstractmethod
    def invoke(self, prompt: str):
        pass