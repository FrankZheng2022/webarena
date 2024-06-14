'''
Functions to query llm (GPT-4) or vlm (GPT-4o)
'''

import autogen
import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vlm(prompt, image_path=None, verbose=True):
    """Call the VLM with a prompt and image path and return the response."""
    if verbose:
        print("Prompt\n", prompt)

    # Getting the base64 string
    base64_image = encode_image(image_path)

    vlm = autogen.OpenAIWrapper(config_list=autogen.config_list_from_json("OAI_CONFIG_LIST_VISION"))
    response = vlm.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }


                ],
            }
        ],
        temperature=0.2
    )
    response = response.choices[0].message.content

    if verbose:
        print(f"vlm's ouput:\n{response}")
    return response

def call_llm(prompt, verbose=True):
    """Call the LLM with a prompt and return the response."""
    if verbose:
        print("Prompt\n", prompt)

    vlm = autogen.OpenAIWrapper(config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
    response = vlm.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ]
    )
    response = response.choices[0].message.content

    if verbose:
        print(response)
    return response

# prompt = "You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them. User's Question: Checkout merge requests assigned to me"
# image_path="./images/test.jpg"
# call_vlm(prompt, image_path, verbose=True)