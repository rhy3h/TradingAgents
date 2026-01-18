import yaml
from pathlib import Path

class PromptLoader:
    def __init__(self, file_path=None):
        if file_path is None:
            # Assumes prompts.yaml is in the parent directory of 'utils'
            current_dir = Path(__file__).parent
            file_path = current_dir.parent / "prompts.yaml"

        with open(file_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

    def get_prompt(self, category, agent, key="prompt_template"):
        """
        Retrieve a prompt or template from the loaded configuration.
        """
        try:
            return self.prompts[category][agent][key]
        except KeyError:
            print(f"Warning: Prompt not found for {category}.{agent}.{key}")
            return ""

    def get_formatted_prompt(self, category, agent, key="prompt_template", **kwargs):
        """
        Retrieve and format a prompt or template.
        """
        template = self.get_prompt(category, agent, key)
        if not template:
            return ""
        return template.format(**kwargs)

# Singleton instance for easy access
_loader = None

def get_prompt_loader():
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader
