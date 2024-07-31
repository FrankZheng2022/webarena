'''
Functions to query llm (GPT-4) or vlm (GPT-4o)
'''

import autogen
import base64
import io
from PIL import Image
import copy
from autogen.code_utils import content_str

def has_image(message):
    if isinstance(message["content"], list):
        for elm in message["content"]:
            if elm.get("type", "") == "image_url":
                return True
    return False


def create_with_images(messages, max_images=1, **kwargs):
    # Clone the messages to give context, but remove old screenshots
    history = []
    n_images = 0
    for m in messages[::-1]:
        # Create a shallow copy
        message = {}
        message.update(m)

        # If there's an image, then consider replacing it with a string
        if has_image(message):
            n_images += 1
            if n_images > max_images:
                message["content"] = content_str(message["content"])

        # Prepend the message -- since we are iterating backwards
        history.insert(0, message)
    return history

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


def call_vlm(messages, max_images=1, tools=None, temperature=0.1, verbose=False):
    # """Call the VLM with a prompt and image path and return the response."""
    
    # messages = create_with_images(messages, max_images=max_images)

    # def replace_image(messages):
    #     for message in messages:
    #         content = message['content']
    #         if isinstance(content, list):
    #             for item in content:
    #                 if item['type'] == 'image_url':
    #                     item['image_url'] = '0'
    #     return messages
    # if verbose:
    #     print("Messages:\n", replace_image(copy.deepcopy(messages)))


    vlm = autogen.OpenAIWrapper(config_list=autogen.config_list_from_json("OAI_CONFIG_LIST_VISION"))
    if tools is None:
        response = vlm.create(
            messages=messages,
            temperature=temperature,
            max_tokens=4096
        )
        response = response.choices[0].message.content
        return response
    else:
        response = vlm.create(
            messages=messages,
            tools=tools,
            tool_choice='auto',
            temperature=temperature,
            max_tokens=4096,
        )
        message = response.choices[0].message
        #print(f"vlm tool calls:\n{message.tool_calls}")
        return message

def call_llm(messages):
    """Call the LLM with a prompt and return the response."""
    llm = autogen.OpenAIWrapper(config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
    response = llm.create(
        messages = messages,
        temperature=0,
        #max_tokens=768,
        top_p=1.0,
    )
    response = response.choices[0].message.content
    return response

# prompt = "You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them. User's Question: Checkout merge requests assigned to me"
# image_path="./images/test.jpg"
# call_vlm(prompt, image_path, verbose=True)