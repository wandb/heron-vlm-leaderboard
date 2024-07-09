import importlib
import os
from typing import Dict, Any

class PluginManager:
    def __init__(self, plugin_dir: str):
        self.plugin_dir = plugin_dir
        self.plugins = {}
        self.load_plugins()

    def load_plugins(self):
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('_adapter.py'):
                try:
                    module_name = f"plugins.{filename[:-3]}"
                    module = importlib.import_module(module_name)
                    if hasattr(module, 'register_plugin'):
                        module.register_plugin(self)
                except  ImportError:
                    continue

    def register_adapter(self, name: str, adapter_class):
        self.plugins[name] = adapter_class

    def get_adapter(self, model_name: str, device: str, config: Dict[str, Any]):
        for adapter_class in self.plugins.values():
            if adapter_class.supports_model(model_name):
                return adapter_class(model_name, device, config)
        raise ValueError(f"No adapter found for model: {model_name}")