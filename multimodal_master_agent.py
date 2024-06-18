from llm_query import call_vlm
from PIL import Image
import autogen.trace as trace
from autogen.trace import node
### A master agent that conditioned on the image input and task instruction, output the high-level plan/instruction
### Question for us: What could be learnable parameters? (Maybe this master agent's System prompt)
class MultimodalMasterAgent:
    
    def __init__(self, intent, start_url, site_description_prompt):
        self.system_prompt = """You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them. """
        self.system_prompt += f"We are visiting the website {start_url} {site_description_prompt}. On this website, please complete the following task:"
        self.system_prompt += f"\n{intent}"
        self.system_prompt += "\nOnce the user has taken the final necessary action to complete the task, and you have fully addressed the initial request, reply with the word TERMINATE."

    def plan(self, image, messages):
        """
            Given the image of the current screen and the user's instruction,
            output a plan to execute
        """
        prompt = self.system_prompt + "\nHistorical Execution traces\n" + messages
        
        instructions = call_vlm(prompt, image)
        return instructions