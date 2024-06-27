# This script runs after GitHub action to check promptflow evaluation
import json
import os
import subprocess
import sys


def check_result(run_name="base_run", cutoff=100.0):
    """
    Check the evaluation result for a given run.

    Args:
        run_name (str, optional): The name of the run to check. Defaults to "base_run".
        cutoff (float, optional): The cutoff value for passing the evaluation. Defaults to 100.0.

    Returns:
        dict: The evaluation result.

    Raises:
        SystemExit: If the evaluation result is below the cutoff value.
    """
    cmd = f"pf run show-metrics -n {run_name}"
    print(cmd)
    # Run cmd and capture output
    result = subprocess.check_output(cmd, shell=True, text=True)
    print(result)
    result = json.loads(result)
    print(result["gpt_groundedness_pass_rate(%)"])
    if result["gpt_groundedness_pass_rate(%)"] < cutoff:
        sys.exit(f"FAILED!!!! Run {run_name} failed with score {result}.")
    else:
        print(f"Run {run_name} passed with score {result}.")
        return result


if __name__ == "__main__":
    check_result()
