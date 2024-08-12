from opto.optimizers import OptoPrimeNewV1, OptoPrimeNewV2
from opto.trace import node, bundle


@bundle(trainable=False)
def f(x):
    return x + 1

@bundle(trainable=False)
def g(x):
    return x * 2

@bundle(trainable=False)
def h(x):
    return x - 1

x = node(1, trainable=True)
x1 = f(x)
x2 = g(x1)
x3 = h(x2)

feedback = """
    1. What is the output of function f?
    2. What is the output of function g?
    3. What is the output of function h?
    Example LLM response:
    {{"reasoning": 'Your reasoning steps',
        "answer": [2, 5, 8]
        "value_check": 'Variable values to check' 
        "suggestion":  'Your suggestions'
    }}
    """



optimizer = OptoPrimeNewV1([x])
optimizer.zero_feedback()
x3.backward(feedback)
response = optimizer.step(x3, feedback)

attempt_n = 0
answer = None
import json
import re
while attempt_n < 2:
    try:
        answer = json.loads(response)["answer"]
        break
    except json.JSONDecodeError:
        # Remove things outside the brackets
        response = re.findall(r"{.*}", response, re.DOTALL)
        if len(response) > 0:
            response = response[0]
        attempt_n += 1
    except Exception:
        attempt_n += 1

print(answer)
assert answer == [2,4,3]