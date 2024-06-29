import json
import os
import re
import subprocess
import time
from multimodal_master_agent import MultimodalMasterAgent
from multimodal_web_surfer_agent import MultimodalWebSurferAgent
from PIL import Image
import io
import base64
from opto.trace import node
from llm_query import call_vlm

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

# Init an environment
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    action2str,
    create_id_based_action,
    create_stop_action,
)
# from evaluation_harness.evaluators import evaluator_router

# Init the environment
env = ScriptBrowserEnv(
    headless=True,
    slow_mo=100,
    observation_type="accessibility_tree",
    current_viewport_only=True,
    viewport_size={"width": 1224, "height": 765},
)


#GITLAB_TASK_LIST = [45, 46, 156]
GITLAB_TASK_LIST = [
    44, 102, 103, 104, 105, 106, 132, 133, 134, 135, 136, 173, 174, 
    175, 176, 177, 205, 206, 207, 293, 294, 295, 296, 297, 349, 350, 
    389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 418, 419, 420, 
    421, 422, 448, 449, 450, 451, 452, 475, 476, 477, 478, 479, 480, 
    481, 482, 483, 484, 485, 522, 567, 568, 569, 570, 590, 591, 592, 
    593, 594, 669, 670, 747, 748, 749, 750, 751, 784, 785, 786, 787, 788
]
# GITLAB_TASK_LIST = [45, 46, 156, 168, 169, 170, 171, 172, 178, 179, 180, 181, 182, 258, 259,\
#                     303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 314, 315, 316, 317, 318, \
#                     339, 340, 341, 342, 343, 357, 411, 412, 413, 414, 415, 416, 417, 441, 442, 443, \
#                     444, 445, 446, 447, 523, 524, 525, 526, 527, 533, 534, 535, 536, 537, 576, 577, \
#                     578, 579, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 736, 742, 743, \
#                     744, 745, 746, 752, 753, 754, 755, 756, 783, 789, 799, 800, 801, 802, 803, 804, \
#                     805, 806, 807, 808, 809, 810, 811]
total_score = 0
for task_id in GITLAB_TASK_LIST:
    config_file = f"config_files/{task_id}.json"

    # set the environment for the current example
    obs, info = env.reset(options={"config_file": config_file})
    #page = node(env.page)
    page = env.page

    with open(config_file, "r") as f:
        config = json.load(f)
        intent = config['intent']

    master_agent = MultimodalMasterAgent(intent, config["start_url"], SITE_DESCRIPTIONS[config["sites"][0]])
    surfer_agent = MultimodalWebSurferAgent(intent, config["start_url"], SITE_DESCRIPTIONS[config["sites"][0]])


    def get_visual_viewport(page):
        try:
            with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
                page.evaluate(fh.read())
        except:
            pass
        return page.evaluate("MultimodalWebSurfer.getVisualViewport();")

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
            messages.append({"role": "user", "content": item[0]})
            messages.append({"role": "user", "content": item[1]})

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



    histories = []
    for i in range(20):

        screenshot = io.BytesIO(page.screenshot())
        img = Image.open(screenshot)
        img = img.resize((1224, 765))
        img.save('./images/screenshot.png')

        instruction = master_agent.plan(histories, './images/screenshot.png')
        if "TERMINATE" in instruction:
            break
        next_obs = surfer_agent.act(histories, page, step=i)
        page = next_obs['page']

        viewport = get_visual_viewport(page)
        percent_visible = int(viewport["height"] * 100 / viewport["scrollHeight"])
        percent_scrolled = int(viewport["pageTop"] * 100 / viewport["scrollHeight"])
        if percent_scrolled < 1:  # Allow some rounding error
            position_text = "at the top of the page"
        elif percent_scrolled + percent_visible >= 99:  # Allow some rounding error
            position_text = "at the bottom of the page"
        else:
            position_text = str(percent_scrolled) + "% down from the top of the page"
        action_description = f"{next_obs['feedback']} Here is a screenshot of [{page.title()}]({page.url}). The viewport shows {percent_visible}% of the webpage, and is positioned {position_text}.".strip()
        histories.append((instruction, action_description))




    final_answer = final_response(intent, histories)
    m = re.search("FINAL ANSWER:(.*)$", final_answer, re.DOTALL)
    if m:
        final_answer = m.group(1).strip()

    ### Evaluation
    from evaluation_harness.evaluators import evaluator_router
    evaluator = evaluator_router(config_file)
    score = evaluator(
        answer=final_answer,
        config_file=config_file,
        page=page,
        client=env.get_page_client(page),
    )
    total_score += score
    print(f'==============TASK:{task_id} SCORE:{score}==================', flush=True)

print(f'==============Average Success Rate:{total_score/len(GITLAB_TASK_LIST)}==============')
