from llm_query import call_vlm
from PIL import Image
from state_of_mark import add_state_of_mark
import os
import re
from browser_env.actions import create_goto_url_action, create_go_back_action, create_scroll_action,\
                                         create_click_action, create_type_action, create_stop_action

MARK_ID_ADDRESS_BAR = 0
MARK_ID_BACK = 1
MARK_ID_RELOAD = 2
MARK_ID_SEARCH_BAR = 3
MARK_ID_PAGE_UP = 4
MARK_ID_PAGE_DOWN = 5

def get_interactive_rects(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getInteractiveRects();")

def get_focused_rect_id(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getFocusedElementId();")

def get_visual_viewport(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getVisualViewport();")

### A master agent that conditioned on the image input and task instruction, output the high-level plan/instruction
### Question for us: What could be learnable parameters? (Maybe this master agent's System prompt)
class MultimodalWebSurferAgent:
    
    def __init__(self):
        return 
    
    ### obs: current observation
    ### page: current webpage
    ### instructions: high-level instructions/plans provided by the users
    def act(self, obs, page, intent, instruction, verbose=True):
        rects = get_interactive_rects(page)
        focused = get_focused_rect_id(page)
        name = rects.get(focused, {}).get("aria-name", "")
        name = f"(and name '{name}') "
        focused_hint = (
            "\nThe "
            + rects.get(focused, {}).get("role", "control")
            + " with ID "
            + str(focused)
            + " "
            + name
            + "currently has the input focus.\n"
        )

        viewport = get_visual_viewport(page)

        som_screenshot, visible_rects = add_state_of_mark(page.screenshot(), rects)
        som_screenshot = som_screenshot.resize((1224, 765))
        som_screenshot.save('images/som_screenshot.png')


        # Include all the static elements
        text_labels = f"""
        {{ "id": {MARK_ID_BACK}, "aria-role": "button", "html_tag": "button", "actions": ["click"], "name": "browser back button" }},
        {{ "id": {MARK_ID_ADDRESS_BAR}, "aria-role": "textbox",   "html_tag": "input, type=text", "actions": ["type"],  "name": "browser address input" }},
        {{ "id": {MARK_ID_SEARCH_BAR}, "aria-role": "searchbox", "html_tag": "input, type=text", "actions": ["type"],  "name": "browser web search input" }},"""

        # We can scroll up
        if viewport["pageTop"] > 5:
            text_labels += f"""
        {{ "id": {MARK_ID_PAGE_UP}, "aria-role": "scrollbar", "html_tag": "button", "actions": ["click", "scroll_up"], "name": "browser scroll up control" }},"""

        # Can scroll down
        if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
            text_labels += f"""
        {{ "id": {MARK_ID_PAGE_DOWN}, "aria-role": "scrollbar", "html_tag": "button", "actions": ["click", "scroll_down"], "name": "browser scroll down control" }},"""

        # Everything visible
        for r in visible_rects:
            if r in rects:
                actions = ["'click'"]
                if rects[r]["role"] in ["textbox", "searchbox", "search"]:
                    actions = ["'type'"]
                # if rects[r]["v-scrollable"]:
                #     actions.append("'scroll_up'")
                #     actions.append("'scroll_down'")
                actions = "[" + ",".join(actions) + "]"

        text_labels += f"""
        {{ "id": {r}, "aria-role": "{rects[r]['role']}", "html_tag": "{rects[r]['tag_name']}", "actions": "{actions}", "name": "{rects[r]['aria-name']}" }},"""


        ### Prepare the final prompt
        text_prompt = f"""
        Consider the following screenshot of a web browser, which is open to the page '{page.url}'. In this screenshot, interactive elements are outlined in bounding boxes of different colors. Each bounding box has a numeric ID label in the same color. Additional information about each visible label is listed below:
        [
        {text_labels}
        ]
        {focused_hint}
        You are to respond to the user's overall intent as well as the detailed instructions by selecting a browser action to perform.
        User's Intent:
        {intent}
        Instructions:
        {instruction}
        Please output the appropriate action in the following format:
        TARGET:   <id of interactive element.>
        ACTION:   <One single action from the element's list of actions>
        ARGUMENT: <The action' argument, if any. For example, the text to type if the action is typing>
        Additionally, you could also choose the following stop actions:
        TARGET: 
        ACTION: stop  
        ARGUMENT: <answer to the user's intent if it is a question, else leave it empty>
        """.strip()

        action_response = call_vlm(text_prompt, image_path='images/som_screenshot.png', verbose=verbose)
        action = self.parse_action(page, rects, action_response)
        return action

    def parse_action(self, page, rects, action_response):

        target = None
        m = re.search(r"TARGET:\s*(\d+)", action_response)
        if m:
            target = m.group(1).strip()

            # # Non-critical. Mainly for pretty logs
            # target_name = rects.get(target, {}).get("aria-name")
            # if target_name:
            #     target_name = target_name.strip()

        action = None
        m = re.search(r"\nACTION:\s*(.*?)\n", action_response)
        if m:
            action = m.group(1).strip().lower()

        m = re.search(r"\nARGUMENT:\s*(.*?)\n", action_response)
        if m:
            argument = m.group(1).strip()

        try:
            if target == str(MARK_ID_ADDRESS_BAR) and argument:
                #action_description = f"I typed '{argument}' into the browser address bar."
                self._log_to_console("goto", arg=argument)
                # Check if the argument starts with a known protocol
                if not argument.startswith(("https://", "http://", "file://")):
                    argument = "https://" + argument
                action = create_goto_url_action(argument)
            elif target == str(MARK_ID_BACK):
                action = create_go_back_action()
            elif target == str(MARK_ID_PAGE_UP):
                action = create_scroll_action(direction='up')
            elif target == str(MARK_ID_PAGE_DOWN):
                action = create_scroll_action(direction='down')
            elif action == "click":
                action = create_click_action(element_id=target)
                target  = page.locator(f"[__elementId='{target}']")
                try:
                    target.wait_for(timeout=100)
                except TimeoutError:
                    raise ValueError("No such element to click!")
                box = target.bounding_box()
                action['pos'] = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            elif action == "type":
                action = create_type_action(text=argument, element_id=target)
            elif action == 'stop':
                action = create_stop_action(argument)
            else:
                raise ValueError
        except ValueError as e:
            raise e



        return action