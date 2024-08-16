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
    A function that adds 1 to x
    """
    return x + 1

@bundle(trainable=False)
def g(x):
    """
    A function that multiplies x by soome constant
    """
    return x * 2

@bundle(trainable=False)
def h(x):
    """
    A function that subtracts 1 from x
    """
    return x - 1

def user_feedback(output, ground_truth=11):
    if output.data == ground_truth:
        return "You get it correctly!"
    else:
        return "The values of your variable is not correct. Output does not match the ground truth value (11)"

def remove_cache():
    cache_dir = os.path.expanduser("./.cache")
    # Check if the directory exists before attempting to remove it
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Removed directory: {cache_dir}")
    else:
        print(f"Directory not found: {cache_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--V2', action='store_true')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--max_optimizer_steps', type=int, default=5)
    args = parser.parse_args()


    objective = dedent(
    """
        You need to change the <value> of the variables in #Variables to improve the output in accordance to #Feedback.
        Find the correct x so that the output of your code matches the ground truth value (11)
    """
    )

    success = 0


    for i in range(args.n_runs):
        x = node(1, trainable=True)

        for i in range(3):
            x1 = f(x)
            x2 = g(x1)
            x3 = h(x2)

            if args.V2:
                optimizer = OptoPrimeNewV2([x])
            else:
                optimizer = OptoPrimeNewV1([x], objective=objective)
            feedback = user_feedback(x3, ground_truth=11)
            optimizer.zero_feedback()
            x3.backward(feedback)
            #try:
            response = optimizer.step(x3, feedback)
            # except:
            #     print('Failure of the optimizer')
            #     remove_cache()
            #     continue

            if feedback == "You get it correctly!":
                success += 1
                print("Success!")
                break
            remove_cache()
    
    print(f"{success} out of {args.n_runs} succeeded")