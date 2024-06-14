from llm_query import call_vlm
from PIL import Image
from state_of_mark import add_state_of_mark
import os
import re
from browser_env.actions import create_goto_url_action, create_go_back_action, create_scroll_action,\
                                create_none_action, create_click_action, create_type_action, create_stop_action
import time
from eval import evaluate

MARK_ID_ADDRESS_BAR = 0
MARK_ID_BACK = 1
MARK_ID_RELOAD = 2
MARK_ID_SEARCH_BAR = 3
MARK_ID_PAGE_UP = 4
MARK_ID_PAGE_DOWN = 5

VIEWPORT_HEIGHT = 900
VIEWPORT_WIDTH = 1440

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

def on_new_page(page):
    page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
    time.sleep(0.2)
    page.add_init_script(path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"))
    page.wait_for_load_state()
    return

def fill_id(page, identifier, value):
    target = page.locator(f"[__elementId='{identifier}']")

    # See if it exists
    try:
        target.wait_for(timeout=100)
    except TimeoutError:
        raise ValueError("No such element.")

    # Fill it
    target.focus()
    target.fill(value)
    page.keyboard.press("Enter")

def scroll_id(page, identifier, direction):
    page.evaluate(
        f"""
    (function() {{
        let elm = document.querySelector("[__elementId='{identifier}']");
        if (elm) {{
            if ("{direction}" == "up") {{
                elm.scrollTop = Math.max(0, elm.scrollTop - elm.clientHeight);
            }}
            else {{
                elm.scrollTop = Math.min(elm.scrollHeight - elm.clientHeight, elm.scrollTop + elm.clientHeight);
            }}
        }}
    }})();
"""
    )

### A master agent that conditioned on the image input and task instruction, output the high-level plan/instruction
### Question for us: What could be learnable parameters? (Maybe this master agent's System prompt)
class MultimodalWebSurferAgent:
    
    def __init__(self):
        return 
    
    ### obs: current observation
    ### page: current webpage
    ### instructions: high-level instructions/plans provided by the users
    def act(self, obs, page, intent, instruction, verbose=True, task_config_file=None):
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
                if rects[r]["v-scrollable"]:
                    actions.append("'scroll_up'")
                    actions.append("'scroll_down'")
                actions = "[" + ",".join(actions) + "]"

                text_labels += f"""
                {{ "id": {r}, "aria-role": "{rects[r]['role']}", "html_tag": "{rects[r]['tag_name']}", "actions": "{actions}", "name": "{rects[r]['aria-name']}" }},"""

        # mask_hint = f"""When identifying the item on the browser, tote that the number (id) of the web element is shown on the top right corner of the masked bounding box instead of being at the bottom. Also, please pay attention to the colors. The color of the element id should be the same as the color of the masked bounding box."""

        ### Prepare the final prompt
        text_prompt = f"""
        Consider the following screenshot of a web browser, which is open to the page '{page.url}'. In this screenshot, interactive elements are outlined in bounding boxes of different colors. Each bounding box has a numeric ID label in the same color. Additional information about each visible label is listed below:
        [
        {text_labels}
        ]
        {focused_hint}
        You are to respond to the user's overall intent as well as the detailed instructions by selecting one next browser action to perform. 
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
        page, action, feedback = self.parse_action(page, rects, action_response, task_config_file=None)
        return page, action, feedback

    def parse_action(self, page, rects, action_response, task_config_file=None):

        target, target_name, argument = None, None, None
        m = re.search(r"TARGET:\s*(\d+)", action_response)
        if m:
            target = m.group(1).strip()

            # # Non-critical. Mainly for pretty logs
            target_name = rects.get(target, {}).get("aria-name")
            if target_name:
                target_name = target_name.strip()

        action = None
        m = re.search(r"\nACTION:\s*(.*?)\n", action_response)
        if m:
            action = m.group(1).strip().lower()

        m = re.search(r"\nARGUMENT:\s*(.*?)\n", action_response)
        if m:
            argument = m.group(1).strip()

        try:
            if target == str(MARK_ID_ADDRESS_BAR) and argument:
                feedback = ""
                if not argument.startswith(("https://", "http://", "file://")):
                    argument = "https://" + argument
                try:
                    action = create_goto_url_action(argument)
                    page.goto(argument)
                    feedback = f"I typed '{argument}' into the browser address bar."
                except ValueError as e:
                    feedback = f"Errors when loading website {argument}, getting error message:{str(e)}"
                    return page, action, feedback
            elif target == str(MARK_ID_BACK):
                action = create_go_back_action()
                feedback = "Going backward one page"
            elif target == str(MARK_ID_PAGE_UP):
                action = create_scroll_action(direction='up')
                feedback = "I scrolled up the page up"
            elif target == str(MARK_ID_PAGE_DOWN):
                action = create_scroll_action(direction='down')
                feedback = "I scrolled down one screen in the browser."
            elif action == "click":
                if target_name:
                    action_description = f"I clicked '{target_name}'."
                else:
                    action_description = "I clicked the control."

                action = create_click_action(element_id=target)
                target  = page.locator(f"[__elementId='{target}']")
                try:
                    target.wait_for(timeout=100)
                    feedback = f"Clicking element {target}"
                except:
                    action = create_none_action()
                    feedback = f"Element {target_name} (id: {target}) doesn't exist! No element to click!"
                    return page, action, feedback
                box = target.bounding_box()
                try:
                    with page.expect_event("popup", timeout=2000) as page_info:
                        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    page = page_info.value
                    on_new_page(page)
                except:
                    action = create_none_action()
                    feedback = "Time out error when clicking element {target}!"
                    return page, action, feedback

            elif action == "type":
                target = page.locator(f"[__elementId='{target}']")
                # See if it exists
                try:
                    target.wait_for(timeout=100)
                    if target_name:
                        feedback = f"I typed '{argument}' into '{target_name}'."
                    else:
                        feedback = f"I input '{argument}'."
                except TimeoutError:
                    action = create_none_action()
                    feedback = f"Element {target} doesn't exist! No element {target} to type!"
                    return page, action, feedback
                # Fill it
                target.focus()
                target.fill(argument if argument else "")
                page.keyboard.press("Enter")
            elif action == "scroll_up":
                if target_name:
                    feedback = f"I scrolled '{target_name}' down."
                else:
                    feedback = "I scrolled the control down."
                try:
                    scroll_id(page, target, "up")
                except:
                    feedback = f"The element {target_name} with id {target} is not scrollable." 
            elif action == "scroll_down":
                if target_name:
                    feedback = f"I scrolled '{target_name}' down."
                else:
                    feedback = "I scrolled the control down."
                try:
                    scroll_id(page, target, "down")
                except:
                    feedback = f"The element {target_name} with id {target} is not scrollable." 

            elif action == 'stop':
                action = create_stop_action(argument)
                feedback = evaluate(argument, task_config_file)
            else:
                raise ValueError
        except ValueError as e:
            action = create_none_action()
            feedback = f"Action you choose is invalid!"



        return page, action, feedback