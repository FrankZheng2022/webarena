from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union
import random
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt
import base64
import autogen
from autogen.code_utils import content_str
import os
from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError
import re
from autogen.token_count_utils import count_token, get_max_token_limit

@dataclass
class DetachedPage:
    url: str
    content: str  # html


def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]


Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]


# Viewport dimensions
VIEWPORT_HEIGHT = 900
VIEWPORT_WIDTH = 1440
MLM_HEIGHT = 765
MLM_WIDTH = 1224

def image_to_data_uri(image):
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

def make_mm_message(text_content, image_content, role="user"):
    return {
        "role": role,
        "content": [
            {"type": "text", "text": text_content},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_uri(image_content),
                },
            },
        ],
    }



def get_target_name(target, rects):
    target_name = rects.get(str(target), {}).get("aria-name")
    if target_name:
        return target_name.strip()
    else:
        return None

def sleep(page, duration):
    page.wait_for_timeout(duration * 1000)


def get_interactive_rects(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getInteractiveRects();")

def get_visual_viewport(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getVisualViewport();")

def get_focused_rect_id(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getFocusedElementId();")

def get_page_metadata(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getPageMetadata();")

def get_page_markdown(page, markdown_converter):
    html = page.evaluate("document.documentElement.outerHTML;")
    res = markdown_converter.convert_stream(io.StringIO(html), file_extension=".html", url=page.url)
    return res.text_content

def on_new_page(page):
    page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
    sleep(page, 0.2)
    page.add_init_script(path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"))
    page.wait_for_load_state()

def back(page):
    page.go_back()

def visit_page(page, url):
    try:
        # Regular webpage
        page.goto(url)
    except Exception as e:
        # Downloaded file
        raise e

def page_down(page):
    page.evaluate(f"window.scrollBy(0, {VIEWPORT_HEIGHT-50});")

def page_up(page):
    page.evaluate(f"window.scrollBy(0, -{VIEWPORT_HEIGHT-50});")

def click_id(page, identifier):
    target = page.locator(f"[__elementId='{identifier}']")

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
        with page.expect_event("popup", timeout=1000) as page_info:
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, delay=10)
            on_new_page(page_info.value)
            return page_info.value
    except TimeoutError:
        return None


def fill_id(page, identifier, value):
    target = page.locator(f"[__elementId='{identifier}']")

    # See if it exists
    try:
        target.wait_for(timeout=100)
        try:
            target.fill(value)
            page.keyboard.press("Enter")
        except PlaywrightError:
            target.press_sequentially(value)
            page.keyboard.press("Enter")
    except TimeoutError:
        raise ValueError("No such element.")


def get_visual_viewport(page):
    try:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            page.evaluate(fh.read())
    except:
        pass
    return page.evaluate("MultimodalWebSurfer.getVisualViewport();")

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

def summarize_page(page, markdown_converter, question=None, token_limit=100000):
    page_markdown = get_page_markdown(page, markdown_converter)

    buffer = ""
    for line in re.split(r"([\r\n]+)", page_markdown):
        tokens = count_token(buffer + line)
        if tokens + 1024 > token_limit:  # Leave room for our summary
            break
        buffer += line

    buffer = buffer.strip()
    if len(buffer) == 0:
        return "Nothing to summarize."

    title = page.url
    try:
        title = page.title()
    except:
        pass

    # Take a screenshot and scale it
    screenshot = page.screenshot()
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
        make_mm_message(prompt, scaled_screenshot),
    )
    scaled_screenshot.close()

    response = call_vlm(messages)
    return str(response)


def get_ocr_text(image):
    
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
        make_mm_message("Please transcribe all visible text on this page, including both main content and the labels of UI elements.", scaled_screenshot),
    ]
    scaled_screenshot.close()

    response = call_vlm(messages)
    return str(response)




def call_vlm(messages, tools=None):
    # """Call the VLM with a prompt and image path and return the response."""
    vlm = autogen.OpenAIWrapper(config_list=autogen.config_list_from_json("OAI_CONFIG_LIST_VISION"))
    if tools is None:
        response = vlm.create(
            messages=messages,
        )
        response = response.choices[0].message.content
        return response
    else:
        response = vlm.create(
            messages=messages,
            tools=tools,
            tool_choice='auto',
        )
        message = response.choices[0].message
        return message
    
TOP_NO_LABEL_ZONE = 20  # Don't print any labels close the top of the page


def add_set_of_mark(screenshot, ROIs):
    if isinstance(screenshot, Image.Image):
        return _add_set_of_mark(screenshot, ROIs)

    if not isinstance(screenshot, io.BufferedIOBase):
        screenshot = io.BytesIO(screenshot)

    image = Image.open(screenshot)
    result = _add_set_of_mark(image, ROIs)
    image.close()
    return result


def _add_set_of_mark(screenshot, ROIs):
    visible_rects = list()
    rects_above = list() # Scroll up to see
    rects_below = list() # Scroll down to see

    fnt = ImageFont.load_default(14)
    base = screenshot.convert("L").convert("RGBA")
    overlay = Image.new("RGBA", base.size)

    draw = ImageDraw.Draw(overlay)
    for r in ROIs:
        for rect in ROIs[r]["rects"]:
            # Empty rectangles
            if not rect:
                continue
            if rect["width"] * rect["height"] == 0:
                continue

            mid = ((rect["right"] + rect["left"]) / 2.0, (rect["top"] + rect["bottom"]) / 2.0)

            if 0 <= mid[0] and mid[0] < base.size[0]:
                if mid[1] < 0:
                    rects_above.append(r)
                elif mid[1] >= base.size[1]:
                    rects_below.append(r)
                else:
                    visible_rects.append(r)
                    _draw_roi(draw, int(r), fnt, rect)

    comp = Image.alpha_composite(base, overlay)
    overlay.close()
    return comp, visible_rects, rects_above, rects_below


def _trim_drawn_text(draw, text, font, max_width):
    buff = ""
    for c in text:
        tmp = buff + c
        bbox = draw.textbbox((0, 0), tmp, font=font, anchor="lt", align="left")
        width = bbox[2] - bbox[0]
        if width > max_width:
            return buff
        buff = tmp
    return buff


def _draw_roi(draw, idx, font, rect):
    color = _color(idx)
    luminance = color[0] * 0.3 + color[1] * 0.59 + color[2] * 0.11
    text_color = (0, 0, 0, 255) if luminance > 90 else (255, 255, 255, 255)

    roi = [(rect["left"], rect["top"]), (rect["right"], rect["bottom"])]

    label_location = (rect["right"], rect["top"])
    label_anchor = "rb"

    if label_location[1] <= TOP_NO_LABEL_ZONE:
        label_location = (rect["right"], rect["bottom"])
        label_anchor = "rt"

    draw.rectangle(roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2)

    bbox = draw.textbbox(label_location, str(idx), font=font, anchor=label_anchor, align="center")
    bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
    draw.rectangle(bbox, fill=color)

    draw.text(label_location, str(idx), fill=text_color, font=font, anchor=label_anchor, align="center")


def _color(identifier):
    rnd = random.Random(int(identifier))
    color = [rnd.randint(0, 255), rnd.randint(125, 255), rnd.randint(0, 50)]
    rnd.shuffle(color)
    color.append(255)
    return tuple(color)