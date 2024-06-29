'''
Functions to query llm (GPT-4) or vlm (GPT-4o)
'''

import autogen
import base64
import io
from PIL import Image
import copy

# Function to encode the image
def encode_image(image_path):
    """
    Image can be a bytes string, a Binary file-like stream, or PIL Image.
    """
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    # image_bytes = image
    # if isinstance(image, Image.Image):
    #     image_buffer = io.BytesIO()
    #     image.save(image_buffer, format="PNG")
    #     image_bytes = image_buffer.getvalue()
    # elif isinstance(image, io.BytesIO):
    #     image_bytes = image_buffer.getvalue()
    # elif isinstance(image, io.BufferedIOBase):
    #     image_bytes = image.read()

    # image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

def call_vlm(messages, temperature=0.1):
    """Call the VLM with a prompt and image path and return the response."""
    
    def replace_image(messages):
        for message in messages:
            content = message['content']
            if isinstance(content, list):
                for item in content:
                    if item['type'] == 'image_url':
                        item['image_url'] = '0'
        return messages
    #print("Messages:\n", replace_image(copy.deepcopy(messages)))

    # Getting the base64 string
    #image_encode = encode_image(image)

    vlm = autogen.OpenAIWrapper(config_list=autogen.config_list_from_json("OAI_CONFIG_LIST_VISION"))
    response = vlm.create(
        messages=messages,
        temperature=temperature
    )
    response = response.choices[0].message.content
    #print(f"vlm's ouput:\n{response}")
    return response

def call_llm(messages):
    """Call the LLM with a prompt and return the response."""
    llm = autogen.OpenAIWrapper(config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
    response = llm.create(
        messages = messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    )
    response = response.choices[0].message.content
    return response

# prompt = "You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them. User's Question: Checkout merge requests assigned to me"
# image_path="./images/test.jpg"
# call_vlm(prompt, image_path, verbose=True)