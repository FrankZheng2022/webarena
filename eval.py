import json
from nltk.tokenize import word_tokenize
from llm_query import call_llm

def clean_answer(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()

def exact_match(ref, pred):
    return clean_answer(ref) == clean_answer(pred)

def fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = '''
    You are a helpful assistant. Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.
    Question: {question}
    Reference answer: {reference}
    All the string 'N/A' that you see is a special sequence that means 'not achievable'
    Student answer: {pred}
    Conclude the judgement by correct/incorrect/partially correct.
    '''
    response = call_llm(message).lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response
        return 1.0
    
def ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message ='''
        You are a helpful assistant.
        Task: {question}\
        Actual unachievable reason: {reference}
        Reported unachievable reason: {pred}
        The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. 
        An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. 
        Determine if the reported reason aligns with the actual reason, even if implicitly. 
        If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'.
        '''

    response = call_llm(message).lower()
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0

def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
    clean_ref = clean_answer(ref)
    clean_pred = clean_answer(pred)
    # tokenize the answer if the ref is a single word
    # prevent false positive (e.g, 0)
    if (
        tokenize
        and len(clean_ref) == 1
        and len(word_tokenize(clean_ref)) == 1
    ):
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref in tok_pred)
    else:
        return float(clean_ref in clean_pred)

def evaluate(pred, task_config_file):
    with open(task_config_file, "r") as f:
        configs = json.load(f)

    score = 1.0
    for approach, value in task_config_file["eval"]["reference_answers"].items():
        match approach:
            case "exact_match":
                score *= exact_match(ref=value, pred=pred)

            case "must_include":
                assert isinstance(value, list)
                for must_value in value:
                    score *= must_include(
                        ref=must_value,
                        pred=pred,
                        tokenize=(len(value) == 1),
                    )
            case "fuzzy_match":
                intent = configs["intent"]
                if value == "N/A":
                    # if the instruction only asks the model to generate N/A when encountering an unachievable task
                    # without more concrete reasons
                    score *= exact_match(ref=value, pred=pred)
                    # if the instruction also asks the model to generate the reason why the task is unachievable
                    # this should be the default as it will prevent false positive N/A`
                    if score != 1:
                        score = 1.0 * ua_match(
                            intent=configs["intent"],
                            ref=configs["eval"]["string_note"],
                            pred=pred,
                        )
                else:
                    assert isinstance(value, list)
                    for reference in value:
                        score *= fuzzy_match(
                            ref=reference, pred=pred, intent=intent
                        )
    return score