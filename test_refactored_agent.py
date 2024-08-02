import json
import os
import re
import subprocess
import time
from agent import MultimodalMasterAgent
from agent import MultimodalWebSurferAgentV2New
from PIL import Image
import io
import base64
#from opto.trace import node
from utils.llm_query import call_vlm
from browser_env import ScriptBrowserEnv, EnvWrapper


def final_response(intent, histories):
    messages = [
                    {
                        "role": "user",
                        "content": f"""Earlier you were asked the following:

                    {intent}

                    Your team then worked diligently to address that request. Here is a transcript of that conversation:""",
                    }
                ]

            # copy them to this context
    for item in histories:
        messages.append({"role": "user", "content": item[0]['content']})
        if item[1] is not None:
            if isinstance(item[1]['content'], str):
                messages.append({"role": "user", "content": item[1]['content']})
            else:
                messages.append({"role": "user", "content": item[1]['content'][0]['text']})

    # ask for the final answer
    messages.append(
        {
            "role": "user",
            "content": f"""Read the above conversation and output a FINAL ANSWER to the original request. The original request is repeated here for convenience:

    {intent}

    To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
    Your FINAL ANSWER should be as few words as possible.
    If the original request was not a question, or you did not find a definitive answer, simply summarize the final state of the page or task as your FINAL ANSWER.""",
            }
        )

    with open('./images/screenshot.png', "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    messages[-1]["content"] = [{"type": "text", "text":messages[-1]["content"]}, 
                            {"type": "image_url", "image_url":{"url":f"data:image/png;base64,{image_base64}"}}, 
                            ]  
    response = call_vlm(messages) 
    # if "finish_reason='content_filter'" in str(response):
    #     raise Exception(str(response))
    return response


SLEEP = 1.5
# set the URLs of each website, we use the demo sites as an example
os.environ[
    "SHOPPING"
] = "http://10.137.68.110:7770"
os.environ[
    "SHOPPING_ADMIN"
] = "http://10.137.68.110:7780/admin"
os.environ[
    "REDDIT"
] = "http://10.137.68.110:9999"
os.environ[
    "GITLAB"
] = "http://10.137.68.110:8023"
os.environ[
    "MAP"
] = "http://10.137.68.110:3000"
os.environ[
    "WIKIPEDIA"
] = "http://10.137.68.110:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site
print("Done setting up URLs")

SITE_DESCRIPTIONS = {
    "reddit": "a Postmill forum populated with a large sample of data crawled from Reddit. Postmill is similar to Reddit, but the UI is distinct, and 'subreddits' begin with /f/ rather than /r/",
    "gitlab": "a Gitlab site populated with various programming projects. Gitlab is similar to GitHub, though the UIs are slightly different",
    "shopping": "an online store built with the Magento open source eCommerce platform",
    "shopping_admin": "the content management admin portal for an online store running the Magento open source eCommerce software",
    "map": "a map for navigation and searching for information about points of interest (POIs) such as institutions or locations"
}

# First, run `python scripts/generate_test_data.py` to generate the config files
p = subprocess.run(
    ["python", "scripts/generate_test_data.py"], capture_output=True
)

# It will generate individual config file for each test example in config_files
assert os.path.exists("config_files/0.json")

# Make sure the URLs in the config files are replaced properly
with open("config_files/0.json", "r") as f:
    config = json.load(f)
    assert os.environ["SHOPPING_ADMIN"] in config["start_url"], (
        os.environ["SHOPPING_ADMIN"],
        config["start_url"],
    )

print("Done generating config files with the correct URLs")

# run bash prepare.sh to save all account cookies, this only needs to be done once
subprocess.run(["bash", "prepare.sh"])
print("Done saving account cookies")




env = EnvWrapper(headless=False)



GITLAB_TASK_LIST = [103]
total_score = 0
for task_id in GITLAB_TASK_LIST:
    
    env.reset(task_id)
    config_file = f"config_files/{task_id}.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    histories = []
    master_agent = MultimodalMasterAgent(config["intent"], config["start_url"], SITE_DESCRIPTIONS[config["sites"][0]], histories)
    surfer_agent = MultimodalWebSurferAgentV2New(config["intent"], config["start_url"], SITE_DESCRIPTIONS[config["sites"][0]], histories)


    
    for i in range(20):

        screenshot = io.BytesIO(env.get_screenshot())
        img = Image.open(screenshot)
        img = img.resize((1224, 765))
        img.save('./images/screenshot.png')
        instruction = master_agent.plan('./images/screenshot.png')
        if "TERMINATE" in instruction:
            break
        tool_name, args = surfer_agent.act(env.get_page(), step=i)
        execution = env.execute_action(tool_name, args)
        print(f"Action Description:{execution['action_description']}")
        surfer_action_summary = execution["surfer_action_summary"]
        histories[-1][1] = {"content": surfer_action_summary}



    ###### Extract Final Answer
    final_answer = final_response(config["intent"], histories)
    m = re.search("FINAL ANSWER:(.*)$", final_answer, re.DOTALL)
    if m:
        final_answer = m.group(1).strip()

    ###### FInal Evaluation
    from evaluation_harness.evaluators import evaluator_router
    config_file = f"config_files/{task_id}.json"
    evaluator = evaluator_router(config_file)
    score = evaluator(
        answer=final_answer,
        config_file=config_file,
        page=env.get_page(),
        client=env._env.get_page_client(env.get_page()),
    )
    total_score += score
    print(f'==============TASK:{task_id} SCORE:{score}==================', flush=True)
    # except:
    #      print()
    #      print(f'==============TASK:{task_id} Evaluation Error==================', flush=True)

print(f'==============Average Success Rate:{total_score/len(GITLAB_TASK_LIST)}==============')