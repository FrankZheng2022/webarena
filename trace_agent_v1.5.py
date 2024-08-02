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
from agent import MultimodalMasterAgent
from agent import MultimodalWebSurferAgentV2New
from PIL import Image
import io
import base64
#from opto.trace import node
from utils.llm_query import call_vlm
from browser_env import ScriptBrowserEnv, TraceEnvWrapper
from browser_env.utils import get_visual_viewport, get_interactive_rects
from utils.set_of_mark_v2 import add_set_of_mark
from utils.tool_definitions import (
    TOOL_VISIT_URL,
    TOOL_WEB_SEARCH,
    TOOL_HISTORY_BACK,
    TOOL_PAGE_UP,
    TOOL_PAGE_DOWN,
    TOOL_CLICK,
    TOOL_TYPE,
    TOOL_SCROLL_ELEMENT_DOWN,
    TOOL_SCROLL_ELEMENT_UP,
    TOOL_SUMMARIZE_PAGE,
    TOOL_READ_PAGE_AND_ANSWER,
    TOOL_SLEEP,
    TOOL_ANSWER
)



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


def get_focused_rect_id(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getFocusedElementId();")

def format_target_list(ids, rects):
    targets = []
    for r in list(set(ids)):
        if r in rects:
            
            # Get the role
            aria_role = rects[r].get("role", "").strip()
            if len(aria_role) == 0:
                aria_role = rects[r].get("tag_name", "").strip()
                
            # Get the name
            aria_name = rects[r].get("aria-name", "").strip()

            # What are the actions?
            actions = ['"click"']
            if rects[r]["role"] in ["textbox", "searchbox", "search"]:
                actions = ['"input_text"']
            actions = "[" + ",".join(actions) + "]"

            targets.append(f'{{"id": {r}, "name": "{aria_name}", "role": "{aria_role}", "tools": {actions} }}')
    return targets



@bundle(trainable=True)
def plan(screenshot_path, user_intent, start_url, site_description_prompt):
    """
        Given the current screenshot of the page, the user_intent, the start url of the webpage and the description of the website,
        return a plan for the agent's future actions.
    """
    system_prompt = """You are a general-purpose AI assistant and can handle many questions but you don't have access to a web browser. However, the user you are talking to does have a browser, and you can see the screen. Provide short direct instructions to them. 
                        Once the user has taken the final necessary action to complete the task, and you have fully addressed the initial request, reply with the word TERMINATE."""
    user_intent   = f"""We are visiting the website {start_url} {site_description_prompt}. On this website, please complete the following task:
                        {user_intent}"""
    

    with open(screenshot_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_url = f"data:image/png;base64,{image_base64}"

    ### Create messages to be sent to OpenAI API 
    messages = [
        {"content": system_prompt, "role": "system"}, 
        {
            "content": [
                {'type': 'text', 'text': user_intent},
                {"type": "image_url", "image_url": {"url": image_url}}
            ], 
            "role": "user"
        }
    ]
    instructions = call_vlm(messages) 

    return instructions

@bundle(trainable=True)
def act(som_screenshot_path, url, plan, user_intent, available_tools_names, start_url, site_description_prompt):
    """
        Given the current screenshot and url of the page, the user_intent, 
        a high-level plan returned by the planner as well, 
        as well as a list of tools currently available,
        return two things: an executable action and its argument. 
        Additionally, you are also given the start_url of the current task and the description of the current website.
        input_text(args): input text: args.get("text_value") into the field id: args.get("input_field_id").
        click(args): click element with id = args.get("target_id").
        answer(args): choose this action if the the plan includes terminate and the user's intent has also been accomplished
                        if the user's intent is a question, answer with args.get("answer"),
                        otherwise, just let args.get("final_answer") to be None
        page_down: scroll the page down
        page_up: scroll the page up
        history_back: go back to the last page
        
        
        You can get access to the following global variables: VISIBLE_TARGETS, OTHER_TARGETS, FOCUSED_HINT
        1. VISIBLE_TARGETS is a string that puts together all the visible interactible elements, their ids and describe the list of possible actions 
        Here is an example of VISIBLE_TARGETS:
              {"id": 146, "name": "The A11Y Project / a11yproject.com", "role": "link", "tools": ["click"] }
              ...
              {"id": 100, "name": "", "role": "searchbox", "tools": ["input_text"] }"
        2. OTHER_TARGETS is another string that puts together all the other interactible elements that you do not see in the viewport. 
        You need to either scroll up or down to see those strings. Format of OTHER_TARGETS is the exact same as VISIBLE_TARGETS.
        3. FOCUSED_HINT is a string that tells you which element has the input focus. 
        Here is an example:
        "The element Starred with ID 132 currently has the input focus."
        If no element has input focus, it will just be an empty string

    """
    global VISIBLE_TARGETS, OTHER_TARGETS, FOCUSED_HINT

    user_intent   = f"""We are visiting the website {start_url} {site_description_prompt}. On this website, please complete the following task:
                        {user_intent}"""
    available_tools = [TOOL_TYPE if tool == 'input_text' else eval("TOOL_" + tool.upper()) for tool in available_tools_names]
    available_tools_names = '\n'.join(available_tools_names)
    text_prompt = f"""
                    Consider the following screenshot of a web browser, which is open to the page '{url}'. In this screenshot, interactive elements are outlined in bounding boxes of different colors. Each bounding box has a numeric ID label in the same color. Additional information about each visible label is listed below:

                    {VISIBLE_TARGETS}{OTHER_TARGETS}{FOCUSED_HINT}You are to respond to the user's most recent request by selecting an appropriate tool the following set (choose answer if TERMINATE appears in the plan and you think the episode has ended.):

                    {available_tools_names}

                    When deciding between tools, consider if the request can be best addressed by:
                    - the contents of the current viewport (in which case actions like clicking links, clicking buttons, or inputting text might be most appropriate) 
                    - contents found elsewhere on the full webpage (in which case actions like scrolling, summarization, or full-page Q&A might be most appropriate)
                    - on some other website entirely (in which case actions like performing a new web search might be the best option)
                    """.strip()
    
    messages = [{"content": user_intent, "role": "user"}, {"content": f"High-level Action Plan:\n{plan}", "role": "assistant"}]
    with open(som_screenshot_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    content = [{"type": "text", "text":text_prompt},
                {"type": "image_url", "image_url":{"url":f"data:image/png;base64,{image_base64}"}}, 
    ]  


    messages.append({"content":content, "role": "user"})
    message = call_vlm(messages,tools=available_tools, verbose=False) 
    action = message.tool_calls[0].function.name
    args = json.loads(message.tool_calls[0].function.arguments)
    return action, args


    






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
    try:
        screenshot_path, som_screenshot_path, done, action_description = env.execute_action(action)  # next_obs, reward, termination, truncation, info
        feedback = user_feedback(action, action_description)
        STEP_COUNT += 1
    except ValueError as e:
        raise ValueError(e)
    return screenshot_path, som_screenshot_path, done, feedback


def user_feedback(action, action_description):
    """
    Provide feedback to the user.
    """
    global STEP_COUNT
    action_description = f"Action Feedback from step{STEP_COUNT}:" + action_description
    if action[0] == "answer":
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


VISIBLE_TARGETS, OTHER_TARGETS, FOCUSED_HINT = None, None, None

def rollout(user_intent, screenshot_path, som_screenshot_path, horizon, planner, actor, env, start_url, site_description_prompt):
    # Reset the env outside
    # Rollout for horizon steps

    global VISIBLE_TARGETS, OTHER_TARGETS, FOCUSED_HINT

    page = env.get_page()
    buffer = defaultdict(list)
    screenshot_list = [screenshot_path.data]
    for _ in range(horizon):
        plan = planner(screenshot_path, user_intent, start_url, site_description_prompt)

        ###################### Definte FOCUSED_HINT ######################
        focused = get_focused_rect_id(page)
        rects = get_interactive_rects(page)
        FOCUSED_HINT = ""
        if focused:
            name = rects.get(focused, {}).get("aria-name", "")
            if name:
                name = f"(and name '{name}') "
            focused_hint = (
                "\nThe "
                + rects.get(focused, {}).get("role", "control")
                + " with ID "
                + focused
                + " "
                + name
                + "currently has the input focus.\n\n"
            )
    
        ###################### Definte VISIBLE_TARGETS and OTHER_TARGETS ########################
        _, visible_rects, rects_above, rects_below = add_set_of_mark(page.screenshot(), rects)

        # Everything visible
        VISIBLE_TARGETS = "\n".join(format_target_list(visible_rects, rects)) + "\n\n"
        OTHER_TARGETS = []
        OTHER_TARGETS.extend(format_target_list(rects_above, rects))
        OTHER_TARGETS.extend(format_target_list(rects_below, rects))

        if len(OTHER_TARGETS) > 0:
            OTHER_TARGETS = "Additional valid interaction targets (not shown) include:\n" + "\n".join(OTHER_TARGETS) + "\n\n"
        else:
            OTHER_TARGETS = ""

        ###################### Definte available_tool_names ######################
        # Define the list of available tools:
        viewport = get_visual_viewport(page)
        # What tools are available?
        tools = [
            TOOL_HISTORY_BACK,
            TOOL_CLICK,
            TOOL_TYPE,
            TOOL_ANSWER
        ]
        # We can scroll up
        if viewport["pageTop"] > 5:
            tools.append(TOOL_PAGE_UP)
        # Can scroll down
        if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
            tools.append(TOOL_PAGE_DOWN)
        available_tools_names = [t["function"]["name"] for t in tools]
        action = actor(som_screenshot_path, page.url, plan, user_intent, available_tools_names, start_url, site_description_prompt)
        #print(f'Actions:{action[0]}, Arguments:{action[1]}')
        
        screenshot_path, som_screenshot_path, done, feedback = step(action)

        buffer['obs'].append(som_screenshot_path)
        buffer["feedback"].append(feedback.data)
        screenshot_list.append(screenshot_path.data)
        #screenshot_list.append(som_screenshot_path.data)
        if done:
            break
    return buffer, done, screenshot_list


### Optimization for multi step
def multi_step(env, start_url, site_description_prompt, user_intent, planner, actor, n_iterations=50, rollout_horizon=1, horizon=20):
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
                screenshot_list = [screenshot_path.data]
                optimizer.objective = f"{optimizer.default_objective}"
            buffer, done, screenshot_list = rollout(user_intent, screenshot_path, som_screenshot_path, rollout_horizon, planner, actor, env, start_url, site_description_prompt)
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
        #print('Computational Graph Saved')
        optimizer.step(verbose=True, screenshot_list=screenshot_list)
        global STEP_COUNT
        print('===========Current Plan Function===========')
        print(planner.parameter.data)
        print('===========Current Act Function===========')
        print(actor.parameter.data, flush=True)

# Need ablation of not tracing step and reset
# Need ablation of ignoring some info in propagated feedback
# Need to test backward across time.

planner = plan
actor = act

TASK_ID = 103
env = TraceEnvWrapper(headless=True)
config_file = f"config_files/{TASK_ID}.json"
with open(config_file, "r") as f:
    config = json.load(f)
STEP_COUNT = 0
multi_step(env, config["start_url"], SITE_DESCRIPTIONS[config["sites"][0]], config["intent"], planner, actor, n_iterations=50, rollout_horizon=2, horizon=20)
