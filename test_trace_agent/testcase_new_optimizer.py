from opto.optimizers import OptoPrimeNewV1, OptoPrimeNewV2
from opto.trace import node, bundle
import json
import re
import shutil
import os
from textwrap import dedent, indent

@bundle(trainable=False)
def f(x):
    """
    A black-box function that does some arithmetic operation on x
    """
    return x + 1

@bundle(trainable=False)
def g(x):
    """
    A black-box function that does some arithmetic operation on x
    """
    return x * 2

@bundle(trainable=False)
def h(x):
    """
    A black-box function that does some arithmetic operation on x
    """
    return x - 1

feedback = ""
# """
#     1. What is the output of function f?
#     2. What is the ou
# tput of function g?
#     3. What is the output of function h?
#     Example LLM response:
#     {{"reasoning": 'Your reasoning steps',
#         "answer": [2, 5, 8]
#         "value_check": 'Variable values to check' 
#         "suggestion":  'Your suggestions'
#     }}
#     """



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--V2', action='store_true')
    parser.add_argument('--n_iterations', type=int, default=1)
    args = parser.parse_args()


    objective = dedent(
    """
        Please answer the following questions:
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
    )

    success = 0
    for i in range(args.n_iterations):

        x = node(1, trainable=True)
        x1 = f(x)
        x2 = g(x1)
        x3 = h(x2)

        if args.V2:
            optimizer = OptoPrimeNewV2([x])
        else:
            optimizer = OptoPrimeNewV1([x], objective=objective)
        optimizer.zero_feedback()
        x3.backward(feedback)
        try:
            response = optimizer.step(x3, feedback)
        except:
            print('Failure of the optimizer')
            # Specify the path to the .cache directory
            cache_dir = os.path.expanduser("~/.cache")

            # Check if the directory exists before attempting to remove it
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"Removed directory: {cache_dir}")
            else:
                print(f"Directory not found: {cache_dir}")
            continue

        attempt_n = 0
        answer = None
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

        # Specify the path to the .cache directory
        cache_dir = os.path.expanduser("~/.cache")

        # Check if the directory exists before attempting to remove it
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Removed directory: {cache_dir}")
        else:
            print(f"Directory not found: {cache_dir}")

        if answer == [2,4,3]:
            success += 1
            print("Success!")
        else:
            print("Failure!")
    print(f'{success} out of {args.n_iterations} is correct!')