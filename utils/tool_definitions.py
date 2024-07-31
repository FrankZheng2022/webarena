TOOL_VISIT_URL = {
    "type": "function",
    "function": {
        "name": "visit_url",
        "description": "Inputs the given url into the browser's address bar, navigating directly to the requested page.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
                "url": {
                    "type": "string",
                    "description": "The URL to visit in the browser.",
                },
            },
            "required": ["reasoning", "url"],
        },
    },
}

TOOL_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Performs a web search on Bing.com with the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
                "query": {
                    "type": "string",
                    "description": "The web search query to use.",
                },
            },
            "required": ["reasoning", "query"],
        },
    },
}

TOOL_HISTORY_BACK = {
    "type": "function",
    "function": {
        "name": "history_back",
        "description": "Navigates back one page in the browser's history. This is equivalent to clicking the browser back button.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

TOOL_PAGE_UP = {
    "type": "function",
    "function": {
        "name": "page_up",
        "description": "Scrolls the entire browser viewport one page UP towards the beginning.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

TOOL_PAGE_DOWN = {
    "type": "function",
    "function": {
        "name": "page_down",
        "description": "Scrolls the entire browser viewport one page DOWN towards the end.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

TOOL_CLICK = {
    "type": "function",
    "function": {
        "name": "click",
        "description": "Clicks the mouse on the target with the given id.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
                "target_id": {
                    "type": "integer",
                    "description": "The numeric id of the target to click.",
                },
            },
            "required": ["reasoning", "target_id"],
        },
    },
}

TOOL_TYPE = {
    "type": "function",
    "function": {
        "name": "input_text",
        "description": "Types the given text value into the specified field.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
                "input_field_id": {
                    "type": "integer",
                    "description": "The numeric id of the input field to receive the text.",
                },
                "text_value": {
                    "type": "string",
                    "description": "The text to type into the input field.",
                },
            },
            "required": ["reasoning", "input_field_id", "text_value"],
        },
    },
}

TOOL_SCROLL_ELEMENT_DOWN = {
    "type": "function",
    "function": {
        "name": "scroll_element_down",
        "description": "Scrolls a given html element (e.g., a div or a menu) DOWN.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
                "target_id": {
                    "type": "integer",
                    "description": "The numeric id of the target to scroll down.",
                },
            },
            "required": ["reasoning", "target_id"],
        },
    },
}

TOOL_SCROLL_ELEMENT_UP = {
    "type": "function",
    "function": {
        "name": "scroll_element_up",
        "description": "Scrolls a given html element (e.g., a div or a menu) UP.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
                "target_id": {
                    "type": "integer",
                    "description": "The numeric id of the target to scroll UP.",
                },
            },
            "required": ["reasoning", "target_id"],
        },
    },
}

TOOL_READ_PAGE_AND_ANSWER = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": "Uses AI to answer a question about the current webpage's content.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
                "question": {
                    "type": "string",
                    "description": "The question to answer.",
                },
            },
            "required": ["reasoning", "question"],
        },
    },
}

TOOL_SUMMARIZE_PAGE = {
    "type": "function",
    "function": {
        "name": "summarize_page",
        "description": "Uses AI to summarize the entire page.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

TOOL_SLEEP = {
    "type": "function",
    "function": {
        "name": "sleep",
        "description": "Wait a short period of time. Call this function if the page has not yet fully loaded, or if it is determined that a small delay would increase the task's chances of success.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
            },
            "required": ["reasoning"],
        },
    },
}


TOOL_ANSWER = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Answer the question that the user has at the end of an episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A short explanation of the reasoning for calling this tool and taking this action.",
                },
            },
            "required": ["reasoning"],
        },
    },
}