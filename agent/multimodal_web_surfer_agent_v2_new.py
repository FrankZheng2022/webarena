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
#import numpy as np
#import easyocr
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
from browser_env.utils import get_visual_viewport, get_interactive_rects
from utils.set_of_mark_v2 import add_set_of_mark
from utils.llm_query import call_vlm
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
)

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

# Size of the image we send to the MLM
# Current values represent a 0.85 scaling to fit within the GPT-4v short-edge constraints (768px)
MLM_HEIGHT = 765
MLM_WIDTH = 1224

def get_focused_rect_id(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getFocusedElementId();")

class MultimodalWebSurferAgentV2New:
    def __init__(self, intent, start_url, site_description_prompt, histories):
        self.user_intent   = f"""We are visiting the website {start_url} {site_description_prompt}. On this website, please complete the following task:
                                {intent}"""
        self.histories = histories
        self._prior_metadata_hash = None
        self._markdown_converter = MarkdownConverter()

    def _target_name(self, target, rects):
        target_name = rects.get(str(target), {}).get("aria-name")
        if target_name:
            return target_name.strip()
        else:
            return None

    def _format_target_list(self, ids, rects):
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


    def act(self, page, step=0):
        """Generate a reply using autogen.oai."""

        # Ask the page for interactive elements, then prepare the state-of-mark screenshot
        rects = get_interactive_rects(page)
        viewport = get_visual_viewport(page)
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(page.screenshot(), rects)

        # What tools are available?
        tools = [
            TOOL_VISIT_URL,
            TOOL_HISTORY_BACK,
            TOOL_CLICK,
            TOOL_TYPE,
            TOOL_SUMMARIZE_PAGE,
            TOOL_READ_PAGE_AND_ANSWER,
            TOOL_SLEEP,
        ]

        # We can scroll up
        if viewport["pageTop"] > 5:
            tools.append(TOOL_PAGE_UP)

        # Can scroll down
        if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
            tools.append(TOOL_PAGE_DOWN)

        # Focus hint
        focused = get_focused_rect_id(page)
        focused_hint = ""
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

        # Everything visible
        visible_targets = "\n".join(self._format_target_list(visible_rects, rects)) + "\n\n"
        # Everything else
        other_targets = []
        other_targets.extend(self._format_target_list(rects_above, rects))
        other_targets.extend(self._format_target_list(rects_below, rects))

        if len(other_targets) > 0:
            other_targets = "Additional valid interaction targets (not shown) include:\n" + "\n".join(other_targets) + "\n\n"
        else:
            other_targets = ""

        tool_names = "\n".join([t["function"]["name"] for t in tools])

        text_prompt = f"""
Consider the following screenshot of a web browser, which is open to the page '{page.url}'. In this screenshot, interactive elements are outlined in bounding boxes of different colors. Each bounding box has a numeric ID label in the same color. Additional information about each visible label is listed below:

{visible_targets}{other_targets}{focused_hint}You are to respond to the user's most recent request by selecting an appropriate tool the following set, or by answering the question directly if possible:

{tool_names}

When deciding between tools, consider if the request can be best addressed by:
    - the contents of the current viewport (in which case actions like clicking links, clicking buttons, or inputting text might be most appropriate) 
    - contents found elsewhere on the full webpage (in which case actions like scrolling, summarization, or full-page Q&A might be most appropriate)
    - on some other website entirely (in which case actions like performing a new web search might be the best option)
""".strip()

        # Scale the screenshot for the MLM, and close the original
        scaled_screenshot = som_screenshot.resize((MLM_WIDTH, MLM_HEIGHT))
        scaled_screenshot.save(f'images/som_screenshot_{step}.png')
        som_screenshot.close()

        # Add the multimodal message and make the request
        
        messages = [{"content":self.user_intent, "role": "assistant"}]
        for item in self.histories:
            if item[1] is not None:
                messages.append({"content":item[0]['content'], "role": "user"})
                messages.append({"content":item[1]['content'][0]['text'], "role": "assistant"})
            else:
                messages.append({"content":item[0]['content'], "role": "user"})

        with open(f'images/som_screenshot_{step}.png', "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        content = [{"type": "text", "text":text_prompt},
                   {"type": "image_url", "image_url":{"url":f"data:image/png;base64,{image_base64}"}}, 
                  ]  
        messages.append({"content":content, "role": "user"})
        message = call_vlm(messages,tools=tools, verbose=False) 
        tool_name = message.tool_calls[0].function.name
        args = json.loads(message.tool_calls[0].function.arguments)
        return tool_name, args