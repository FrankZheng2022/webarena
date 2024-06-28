from llm_query import call_vlm
from PIL import Image
import base64
# import opto.trace as trace
# from opto.trace import node
### A master agent that conditioned on the image input and task instruction, output the high-level plan/instruction
### Question for us: What could be learnable parameters? (Maybe this master agent's System prompt)
class MultimodalMasterAgent:
    
    def __init__(self, intent, start_url, site_description_prompt):
        self.system_prompt = """You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them. 
                                Once the user has taken the final necessary action to complete the task, and you have fully addressed the initial request, reply with the word TERMINATE."""
        self.user_intent   = f"""We are visiting the website {start_url} {site_description_prompt}. On this website, please complete the following task:
                                {intent}"""
    def plan(self, histories, image):
        """
            Given the image (path) of the current screen,
            output a plan to execute
        """
        self.system_prompt
        messages = [{"content":self.system_prompt, "role": "system"}, {"content":self.user_intent, "role": "user"}]
        for item in histories:
            messages.append({"content":item[0], "role": "assistant"})
            messages.append({"content":item[1], "role": "user"})

        with open(image, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        messages[-1]["content"] = [{"type": "text", "text":messages[-1]["content"]}, 
                                   {"type": "image_url", "image_url":{"url":f"data:image/png;base64,{image_base64}"}}, 
                                  ]  
        instructions = call_vlm(messages) 
        return instructions


    