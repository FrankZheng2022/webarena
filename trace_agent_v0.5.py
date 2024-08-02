from opto import trace
from opto.trace import node, bundle, ExecutionError, Node
from opto.optimizers import OptoPrime
from collections import defaultdict
import copy
import pickle
import json
import os
import re
import subprocess
import time
from PIL import Image
import io
import base64
#from opto.trace import node
from utils.llm_query import call_vlm
from browser_env import TraceEnvWrapper


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

@bundle(trainable=True)
def plan(screenshot_path, user_intent):
    """
        Given the current screenshot of the page and the user_intent, return a plan for the agent's future actions.
        To use GPT-4o API, use the following function:
            call_vlm(messages)
        Here is an example message list:
        ```
            messages = [
                {"role":"system", "content": "You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them."}
                {"role":"user", "content": [
                    {"type": "text", "text":"Abishek wants to check my dotfile configurations. Please invite him to the repo as a guest"},
                    {"type": "image_url", "image_url":{"url": image_url}}
                ]}
            ]
        ```
        To convert the screenshot into image_url here, use the following code:
        ```
        import base64
        with open(screenshot_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"
        ```
    """
    return None

@bundle(trainable=True)
def act(som_screenshot_path, plan, user_intent):
    """
        Given the current screenshot of the page, the user_intent, 
        and a high-level plan as well, return an executable action as well as its argument.
        Here is a list of a set of actions: 
            input_text(args): input text: args.get("text_value") into the field id: args.get("input_field_id").
            click(args): click element with id = args.get("target_id").
            answer(args): choose this action if the user's intent has also been accomplished
                          if the user's intent is a question, answer with args.get("answer"),
                          otherwise, just let args.get("final_answer") to be None
            page_down: scroll the page down, 
            page_up: scroll the page up:
            history_back: go back to the last page, 
        Some examples output of the function of this function:
            "input_text", {"text_value": "test", "input_field_id": 53}
            "page_up", None

        You are privided with a path to the screenshot, where each interactive element is outlined in bounding boxes of different colors. 
        Each bounding box has a numeric ID label in the same color.    

        You are also allowed to use GPT-4o API. To use GPT-4o API, use the following function:
            call_vlm(messages)
            
        Here is an example message list:
        ```
            messages = [
                {"role":"system", "content": "You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them."}
                {"role":"user", "content": [
                    {"type": "text", "text":"Abishek wants to check my dotfile configurations. Please invite him to the repo as a guest"},
                    {"type": "image_url", "image_url":{"url": image_url}}
                ]}
            ]
        ```

        To convert the screenshot into image_url here, use the following code:
        ```
        import base64
        with open(som_screenshot_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"
        ```
    """
    return None, None




@bundle(trainable=False, overwrite_python_recursion=False)
def reset():
    """
    Reset the environment and return the initial screenshot.
    """
    return env.reset(TASK_ID)  


@bundle(trainable=False, overwrite_python_recursion=False)
def step(action):
    """
    Take action in the environment and return the screenshot (path) of next page, screenshot with set of mark for next page.
    """
    global STEP_COUNT
    STEP_COUNT += 1
    try:
        screenshot_path, som_screenshot_path, done, action_description = env.execute_action(action)  # next_obs, reward, termination, truncation, info
        feedback = user_feedback(screenshot_path, action, action_description)
    except ValueError as e:
        raise ValueError(e)
    return node(screenshot_path), node(som_screenshot_path), done, feedback


def user_feedback(screenshot_path, action, action_description):
    """
    Provide feedback to the user.
    """
    global STEP_COUNT
    action_description = f"Action Feedback from step{STEP_COUNT}:" + action_description
    if action[0]== "answer":
        final_answer = action[1].get("final_answer")
        from evaluation_harness.evaluators import evaluator_router
        config_file = f"config_files/{TASK_ID}.json"
        evaluator = evaluator_router(config_file)
        score = evaluator(
            answer=final_answer,
            config_file=config_file,
            page=env.get_page(),
            client=env._env.get_page_client(env.get_page()),
        )
        if score == 0.:
            feedback = action_description + "\nThe episode has finished, and your failed to follow user's intent"
        else:
            feedback = action_description + "\nThe episode has finished, and your have successfully followed user's intent"
    else:
        feedback = action_description
    return feedback


def rollout(user_intent, screenshot_path, som_screenshot_path, horizon, planner, actor):
    # Reset the env outside
    # Rollout for horizon steps

    buffer = defaultdict(list)
    screenshot_list = [screenshot_path.data]
    for _ in range(horizon):
        plan = planner(screenshot_path, user_intent)
        action = actor(som_screenshot_path, plan, user_intent)
        screenshot_path, som_screenshot_path, done, feedback = step(action)
        buffer['obs'].append(som_screenshot_path)
        buffer["feedback"].append(feedback.data)
        screenshot_list.append(screenshot_path.data)
        #screenshot_list.append(som_screenshot_path.data)
        if done:
            break
    return buffer, done, screenshot_list


### Optimization for multi step
def multi_step(user_intent, planner, actor, n_iterations=50, rollout_horizon=1, horizon=20):
    optimizer = OptoPrime(planner.parameters() + 
                                  actor.parameters()
                                  )
    data = list()
    traj = defaultdict(list)
    done = True
    for i in range(n_iterations):  # iterations
        error = None
        try:  # Trace the rollout; detach init_obs to avoid back-propagating across time.
            if done:
                traj = defaultdict(list)
                data.append(traj)
                screenshot_path, som_screenshot_path = reset()
                optimizer.objective = f"{optimizer.default_objective}"
                screenshot_list = [screenshot_path.data, som_screenshot_path.data]
            buffer, done, screenshot_list = rollout(user_intent, screenshot_path, som_screenshot_path, rollout_horizon, planner, actor)

        except ExecutionError as e:
            error = e

        if error is None:
            feedback = "\n".join(buffer["feedback"])
            target = buffer["obs"][-1]  # last observation
        else:
            feedback = str(error)
            target = error.exception_node

        # Optimization
        optimizer.zero_feedback()
        graph = optimizer.backward(target, feedback, retain_graph=True, visualize=True) 
        graph.render(filename=f'images/trace_images/computational_graph_{i}', format='png')
        optimizer.step(verbose=True, screenshot_list=screenshot_list)
        global STEP_COUNT
        print('===========Current Plan Function===========')
        print(planner.parameter.data)
        print('===========Current Act Function===========')
        print(actor.parameter.data, flush=True)

# Need ablation of not tracing step and reset
# Need ablation of ignoring some info in propagated feedback

planner = plan
actor = act

TASK_ID = 44
env = TraceEnvWrapper(headless=True)
config_file = f"config_files/{TASK_ID}.json"
with open(config_file, "r") as f:
    config = json.load(f)
STEP_COUNT = 0
multi_step(config["intent"], planner, actor, n_iterations=50, rollout_horizon=3, horizon=20)
