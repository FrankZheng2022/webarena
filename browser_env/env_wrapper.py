# ruff: noqa: E722
import re
import time
import os
import json
import io
import base64
import pathlib
import hashlib
import traceback
import json
from PIL import Image
from urllib.parse import urlparse, quote, quote_plus, unquote, urlunparse, parse_qs
from typing import Any, Dict, List, Optional, Union, Callable, Literal, Tuple
from typing_extensions import Annotated
from playwright.sync_api import sync_playwright
from playwright._impl._errors import TimeoutError
from playwright._impl._errors import Error as PlaywrightError
from autogen.browser_utils.mdconvert import MarkdownConverter, UnsupportedFormatException, FileConversionException
from autogen.token_count_utils import count_token, get_max_token_limit
#from playwright.sync_api import sync_playwright, TimeoutError
from .utils import *
from browser_env import ScriptBrowserEnv

# Viewport dimensions
VIEWPORT_HEIGHT = 900
VIEWPORT_WIDTH = 1440

class EnvWrapper:
    def __init__(self, headless=False, viewport_height=VIEWPORT_HEIGHT, viewport_width=VIEWPORT_WIDTH):
        self._env = ScriptBrowserEnv(
            headless=headless,
            slow_mo=100,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": viewport_width, "height": viewport_height},
        )
        self._page = None
        self._obs  = None
        self._config_file = None
        self._task_id = None
        self._prior_metadata_hash = None

    def reset(self, task_id, config_file_path="./config_files"):
        self._task_id = task_id
        self._config_file = f"{config_file_path}/{self._task_id}.json"
        # set the environment for the current example
        self._env.reset(options={"config_file": self._config_file})
        self._env.context.set_default_timeout(60000) # One minute
        self._page = self._env.page
        screenshot = io.BytesIO(self._page.screenshot())
        return screenshot
    
    def get_page(self):
        return self._page
    
    def get_screenshot(self):
        return self._page.screenshot()
    
    def get_set_of_mark_obs(self):
        rects = get_interactive_rects(self._page)
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(page.screenshot(), rects)
        return som_screenshot, visible_rects, rects_above, rects_below

    def get_visual_viewport(self):
        viewport = get_visual_viewport(self._page)
        return viewport
    
    def get_user_intent(self):
        with open(self._config_file, "r") as f:
            config = json.load(f)
            user_intent = config['intent']
        return user_intent
    
    def get_start_url(self):
        with open(self._config_file, "r") as f:
            config = json.load(f)
            user_intent = config['start_url']
        return user_intent

    def execute_action(self, message):
        rects = get_interactive_rects(self._page)
        
        #### Execute the action
        action_description = ""
        try:
            if message.tool_calls:
                # We will only call one tool
                name = message.tool_calls[0].function.name
                args = json.loads(message.tool_calls[0].function.arguments)

                if name == "visit_url":
                    url = args.get("url")
                    action_description = f"I typed '{url}' into the browser address bar."
                    # Check if the argument starts with a known protocol
                    if url.startswith(("https://", "http://", "file://", "about:")):
                        visit_page(self._page, url)
                    # If the argument contains a space, treat it as a search query
                    elif " " in url:
                        visit_page(self._page, f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH")
                    # Otherwise, prefix with https://
                    else:
                        visit_page(self._page, "https://" + url)
                    self._prior_metadata_hash = None

                elif name == "history_back":
                    action_description = "I clicked the browser back button."
                    back(self._page)

                elif name == "web_search":
                    query = args.get("query")
                    action_description = f"I typed '{query}' into the browser search bar."
                    visit_page(self._page, f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH")
                    self._prior_metadata_hash = None

                elif name == "page_up":
                    action_description = "I scrolled up one page in the browser."
                    page_up(self._page)

                elif name == "page_down":
                    action_description = "I scrolled down one page in the browser."
                    page_down(self._page)

                elif name == "click":
                    target_id = str(args.get("target_id"))
                    target_name = get_target_name(target_id, rects)
                    if target_name:
                        action_description = f"I clicked '{target_name}'."
                    else:
                        action_description = "I clicked the control."
                    new_page = click_id(self._page, target_id) 
                    if new_page is not None:
                        self._page = new_page
                        self._prior_metadata_hash = None


                elif name == "input_text":
                    input_field_id = str(args.get("input_field_id"))
                    text_value = str(args.get("text_value"))
                    input_field_name = get_target_name(input_field_id, rects)
                    if input_field_name:
                        action_description = f"I typed '{text_value}' into '{input_field_name}'."
                    else:
                        action_description = f"I input '{text_value}'."
                    fill_id(self._page, input_field_id, text_value)

                elif name == "scroll_element_up":
                    target_id = str(args.get("target_id"))
                    target_name = get_target_name(target_id, rects)

                    if target_name:
                        action_description = f"I scrolled '{target_name}' up."
                    else:
                        action_description = "I scrolled the control up."

                    scroll_id(self._page, target_id, "up")

                elif name == "scroll_element_down":
                    target_id = str(args.get("target_id"))
                    target_name = target_name(target_id, rects)

                    if target_name:
                        action_description = f"I scrolled '{target_name}' down."
                    else:
                        action_description = "I scrolled the control down."

                    scroll_id(self._page, target_id, "down")

                elif name == "answer_question":
                    question = str(args.get("question"))
                    action_description = self._summarize_page(self._page, MarkdownConverter(), question=question)

                elif name == "summarize_page":
                    action_description = self._summarize_page(self._page, MarkdownConverter())

                elif name == "sleep":
                    action_description = "I am waiting a short period of time before taking further action."
                    sleep(self._page, 3) # There's a 2s sleep below too

                else:
                    raise ValueError(f"Unknown tool '{name}'. Please choose from:\n\n{tool_names}")

        except ValueError as e:
            action_description = f'I encountered an error when executing action {message}, Error:{str(e)}'

        self._page.wait_for_load_state()
        sleep(self._page, 3)

        #### Handle metadata of the new page
        page_metadata = json.dumps(get_page_metadata(self._page), indent=4)
        metadata_hash = hashlib.md5(page_metadata.encode("utf-8")).hexdigest()
        if metadata_hash != self._prior_metadata_hash:
            page_metadata = "\nThe following metadata was extracted from the webpage:\n\n" + page_metadata.strip() + "\n"
        else:
            page_metadata = ""
        self._prior_metadata_hash = metadata_hash

        ### Describe the viewport of the new page in words
        viewport = self.get_visual_viewport()
        percent_visible = int(viewport["height"] * 100 / viewport["scrollHeight"])
        percent_scrolled = int(viewport["pageTop"] * 100 / viewport["scrollHeight"])
        if percent_scrolled < 1:  # Allow some rounding error
            position_text = "at the top of the page"
        elif percent_scrolled + percent_visible >= 99:  # Allow some rounding error
            position_text = "at the bottom of the page"
        else:
            position_text = str(percent_scrolled) + "% down from the top of the page"
        
        new_screenshot = self._page.screenshot()
        ocr_text = get_ocr_text(new_screenshot)
        message_content = message.content or ""
        surfer_action_summary = f"{message_content}\n\n{action_description}\n\nHere is a screenshot of [{self._page.title()}]({self._page.url}). The viewport shows {percent_visible}% of the webpage, and is positioned {position_text}.{page_metadata}\nAutomatic OCR of the page screenshot has detected the following text:\n\n{ocr_text}".strip()
        return {"page":self._page, "screenshot": new_screenshot, "action_description": action_description, "surfer_action_summary": surfer_action_summary}

