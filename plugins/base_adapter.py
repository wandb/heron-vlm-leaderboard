from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List

class BaseAdapter(ABC):
    dependencies: ClassVar[List[str]] = []

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

    @classmethod
    @abstractmethod
    def supports_model(cls, model_name: str) -> bool:
        pass

    @abstractmethod
    async def generate_response(self, question: str, image_path: str) -> str:
        pass

    @abstractmethod
    async def verify(self) -> bool:
        pass

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {}