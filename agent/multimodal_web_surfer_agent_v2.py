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

# Viewport dimensions
VIEWPORT_HEIGHT = 900
VIEWPORT_WIDTH = 1440

# Size of the image we send to the MLM
# Current values represent a 0.85 scaling to fit within the GPT-4v short-edge constraints (768px)
MLM_HEIGHT = 765
MLM_WIDTH = 1224



class MultimodalWebSurferAgentV2:
    def __init__(self, intent, start_url, site_description_prompt, histories):
        self.user_intent   = f"""We are visiting the website {start_url} {site_description_prompt}. On this website, please complete the following task:
                                {intent}"""
        self.histories = histories

        self._prior_metadata_hash = None

        self._markdown_converter = MarkdownConverter()

        # # Create the page
        # self._context.set_default_timeout(60000) # One minute
        # self._page = self._context.new_page()
        # self._page.route(lambda x: True, self._route_handler)
        # self._page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        # self._page.add_init_script(path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"))
        # self._page.goto(self.start_page)
        # self._page.wait_for_load_state()
        # self._sleep(1)


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


    def _sleep(self, duration):
        self._page.wait_for_timeout(duration * 1000)


    def act(self, page, step=0):
        """Generate a reply using autogen.oai."""
        self._page = page

        # # Clone the messages to give context, removing old screenshots
        # history = []
        # for m in messages:
        #     message = {}
        #     message.update(m)
        #     message["content"] = content_str(message["content"])
        #     history.append(message)

        # Ask the page for interactive elements, then prepare the state-of-mark screenshot
        rects = self._get_interactive_rects()
        viewport = self._get_visual_viewport()
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(self._page.screenshot(), rects)

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

        # # Can we reach Bing to search?
        # if self._navigation_allow_list("https://www.bing.com/"):
        #     tools.append(TOOL_WEB_SEARCH)

        # We can scroll up
        if viewport["pageTop"] > 5:
            tools.append(TOOL_PAGE_UP)

        # Can scroll down
        if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
            tools.append(TOOL_PAGE_DOWN)

        # Focus hint
        focused = self._get_focused_rect_id()
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


        # If there are scrollable elements, then add the corresponding tools
        #has_scrollable_elements = False
        #if has_scrollable_elements:
        #    tools.append(TOOL_SCROLL_ELEMENT_UP)
        #    tools.append(TOOL_SCROLL_ELEMENT_DOWN)

        tool_names = "\n".join([t["function"]["name"] for t in tools])

        text_prompt = f"""
Consider the following screenshot of a web browser, which is open to the page '{self._page.url}'. In this screenshot, interactive elements are outlined in bounding boxes of different colors. Each bounding box has a numeric ID label in the same color. Additional information about each visible label is listed below:

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
        message = call_vlm(messages,tools=tools) 
        print(f'Message from the surfer agent:{message}', flush=True)
        
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
                        self._visit_page(url)
                    # If the argument contains a space, treat it as a search query
                    elif " " in url:
                        self._visit_page(f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH")
                    # Otherwise, prefix with https://
                    else:
                        self._visit_page("https://" + url)
                    self._prior_metadata_hash = None

                elif name == "history_back":
                    action_description = "I clicked the browser back button."
                    self._back()

                elif name == "web_search":
                    query = args.get("query")
                    action_description = f"I typed '{query}' into the browser search bar."
                    self._visit_page(f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH")

                elif name == "page_up":
                    action_description = "I scrolled up one page in the browser."
                    self._page_up()

                elif name == "page_down":
                    action_description = "I scrolled down one page in the browser."
                    self._page_down()

                elif name == "click":
                    target_id = str(args.get("target_id"))
                    target_name = self._target_name(target_id, rects)
                    if target_name:
                        action_description = f"I clicked '{target_name}'."
                    else:
                        action_description = "I clicked the control."
                    self._click_id(target_id)

                elif name == "input_text":
                    input_field_id = str(args.get("input_field_id"))
                    text_value = str(args.get("text_value"))
                    input_field_name = self._target_name(input_field_id, rects)
                    if input_field_name:
                        action_description = f"I typed '{text_value}' into '{input_field_name}'."
                    else:
                        action_description = f"I input '{text_value}'."
                    self._fill_id(input_field_id, text_value)

                elif name == "scroll_element_up":
                    target_id = str(args.get("target_id"))
                    target_name = self._target_name(target_id, rects)

                    if target_name:
                        action_description = f"I scrolled '{target_name}' up."
                    else:
                        action_description = "I scrolled the control up."

                    self._scroll_id(target_id, "up")

                elif name == "scroll_element_down":
                    target_id = str(args.get("target_id"))
                    target_name = self._target_name(target_id, rects)

                    if target_name:
                        action_description = f"I scrolled '{target_name}' down."
                    else:
                        action_description = "I scrolled the control down."

                    self._scroll_id(target_id, "down")

                elif name == "answer_question":
                    question = str(args.get("question"))
                    action_description = self._summarize_page(question=question)

                elif name == "summarize_page":
                    action_description = self._summarize_page()

                elif name == "sleep":
                    action_description = "I am waiting a short period of time before taking further action."
                    self._sleep(3) # There's a 2s sleep below too

                else:
                    raise ValueError(f"Unknown tool '{name}'. Please choose from:\n\n{tool_names}")

        except ValueError as e:
            action_description = f'I encountered an error when executing action {message}, Error:{str(e)}'

        self._page.wait_for_load_state()
        self._sleep(3)

        # Handle metadata
        page_metadata = json.dumps(self._get_page_metadata(), indent=4)
        metadata_hash = hashlib.md5(page_metadata.encode("utf-8")).hexdigest()
        if metadata_hash != self._prior_metadata_hash:
            page_metadata = "\nThe following metadata was extracted from the webpage:\n\n" + page_metadata.strip() + "\n"
        else:
            page_metadata = ""
        self._prior_metadata_hash = metadata_hash

        # Describe the viewport of the new page in words
        viewport = self._get_visual_viewport()
        percent_visible = int(viewport["height"] * 100 / viewport["scrollHeight"])
        percent_scrolled = int(viewport["pageTop"] * 100 / viewport["scrollHeight"])
        if percent_scrolled < 1:  # Allow some rounding error
            position_text = "at the top of the page"
        elif percent_scrolled + percent_visible >= 99:  # Allow some rounding error
            position_text = "at the bottom of the page"
        else:
            position_text = str(percent_scrolled) + "% down from the top of the page"

        new_screenshot = self._page.screenshot()


        ocr_text = self._get_ocr_text(new_screenshot)
        # Return the complete observation
        message_content = message.content or ""
        surfer_action_summary = f"{message_content}\n\n{action_description}\n\nHere is a screenshot of [{self._page.title()}]({self._page.url}). The viewport shows {percent_visible}% of the webpage, and is positioned {position_text}.{page_metadata}\nAutomatic OCR of the page screenshot has detected the following text:\n\n{ocr_text}".strip()
        self.histories[-1][1] = {"content": surfer_action_summary}
        return {"page":self._page}

    def _image_to_data_uri(self, image):
        """
        Image can be a bytes string, a Binary file-like stream, or PIL Image.
        """
        image_bytes = image
        if isinstance(image, Image.Image):
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()
        elif isinstance(image, io.BytesIO):
            image_bytes = image_buffer.getvalue()
        elif isinstance(image, io.BufferedIOBase):
            image_bytes = image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"

    def _make_mm_message(self, text_content, image_content, role="user"):
        return {
            "role": role,
            "content": [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_data_uri(image_content),
                    },
                },
            ],
        }

    def _get_interactive_rects(self):
        try:
            with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
                self._page.evaluate(fh.read())
        except:
            pass
        return self._page.evaluate("MultimodalWebSurfer.getInteractiveRects();")

    def _get_visual_viewport(self):
        try:
            with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
                self._page.evaluate(fh.read())
        except:
            pass
        return self._page.evaluate("MultimodalWebSurfer.getVisualViewport();")

    def _get_focused_rect_id(self):
        try:
            with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
                self._page.evaluate(fh.read())
        except:
            pass
        return self._page.evaluate("MultimodalWebSurfer.getFocusedElementId();")

    def _get_page_metadata(self):
        try:
            with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
                self._page.evaluate(fh.read())
        except:
            pass
        return self._page.evaluate("MultimodalWebSurfer.getPageMetadata();")

    def _get_page_markdown(self):
        html = self._page.evaluate("document.documentElement.outerHTML;")
        res = self._markdown_converter.convert_stream(io.StringIO(html), file_extension=".html", url=self._page.url)
        return res.text_content

    def _on_new_page(self, page):
        self._page = page
        self._page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        self._sleep(0.2)
        self._prior_metadata_hash = None
        self._page.add_init_script(path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"))
        self._page.wait_for_load_state()

    def _back(self):
        self._page.go_back()

    def _visit_page(self, url):
        try:
            # Regular webpage
            self._page.goto(url)
            self._prior_metadata_hash = None
        except Exception as e:
            # Downloaded file
            raise e

    def _page_down(self):
        self._page.evaluate(f"window.scrollBy(0, {VIEWPORT_HEIGHT-50});")

    def _page_up(self):
        self._page.evaluate(f"window.scrollBy(0, -{VIEWPORT_HEIGHT-50});")

    def _click_id(self, identifier):
        target = self._page.locator(f"[__elementId='{identifier}']")

        # See if it exists
        try:
            target.wait_for(timeout=100)
        except TimeoutError:
            raise ValueError("No such element.")

        # Click it
        target.scroll_into_view_if_needed()
        box = target.bounding_box()
        try:
            # Give it a chance to open a new page
            with self._page.expect_event("popup", timeout=1000) as page_info:
                self._page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, delay=10)
                self._on_new_page(page_info.value)
                self._prior_metadata_hash = None
        except TimeoutError:
            pass


    def _fill_id(self, identifier, value):
        target = self._page.locator(f"[__elementId='{identifier}']")

        # See if it exists
        try:
            target.wait_for(timeout=100)
            try:
                target.fill(value)
                self._page.keyboard.press("Enter")
            except PlaywrightError:
                target.press_sequentially(value)
                self._page.keyboard.press("Enter")
        except TimeoutError:
            raise ValueError("No such element.")


    def _scroll_id(self, identifier, direction):
        self._page.evaluate(
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

    def _summarize_page(self, question=None, token_limit=100000):
        page_markdown = self._get_page_markdown()

        buffer = ""
        for line in re.split(r"([\r\n]+)", page_markdown):
            tokens = count_token(buffer + line)
            if tokens + 1024 > token_limit:  # Leave room for our summary
                break
            buffer += line

        buffer = buffer.strip()
        if len(buffer) == 0:
            return "Nothing to summarize."

        title = self._page.url
        try:
            title = self._page.title()
        except:
            pass

        # Take a screenshot and scale it
        screenshot = self._page.screenshot()
        if not isinstance(screenshot, io.BufferedIOBase):
            screenshot = io.BytesIO(screenshot)
        screenshot = Image.open(screenshot)
        scaled_screenshot = screenshot.resize((MLM_WIDTH, MLM_HEIGHT))
        screenshot.close()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can summarize long documents to answer question.",
            }
        ]

        prompt = f"We are visiting the webpage '{title}'. Its full-text contents are pasted below, along with a screenshot of the page's current viewport."
        if question is not None:
            prompt += (
                f" Please summarize the webpage into one or two paragraphs with respect to '{question}':\n\n{buffer}"
            )
        else:
            prompt += f" Please summarize the webpage into one or two paragraphs:\n\n{buffer}"

        messages.append(
            self._make_mm_message(prompt, scaled_screenshot),
        )
        scaled_screenshot.close()

        response = call_vlm(messages)
        return str(response)


    def _get_ocr_text(self, image):
        
        scaled_screenshot = None
        if isinstance(image, Image.Image):
            scaled_screenshot = image.resize((MLM_WIDTH, MLM_HEIGHT))
        else:
            pil_image = None
            if not isinstance(image, io.BufferedIOBase):
                pil_image = Image.open(io.BytesIO(image))
            else:
                pil_image = Image.open(image)
            scaled_screenshot = pil_image.resize((MLM_WIDTH, MLM_HEIGHT))
            pil_image.close()

        messages = [
            self._make_mm_message("Please transcribe all visible text on this page, including both main content and the labels of UI elements.", scaled_screenshot),
        ]
        scaled_screenshot.close()

        response = call_vlm(messages)
        return str(response)
