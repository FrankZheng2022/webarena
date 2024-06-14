from llm_query import call_vlm
from PIL import Image
### A master agent that conditioned on the image input and task instruction, output the high-level plan/instruction
### Question for us: What could be learnable parameters? (Maybe this master agent's System prompt)
class MultimodalMasterAgent:
    
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
    
    def plan(self, image_obs, image_path='./images/screenshot.png'):
        
        #img = Image.fromarray(image_obs)
        #img.save(image_path)
        instructions = call_vlm(self.system_prompt, image_path)
        return instructions