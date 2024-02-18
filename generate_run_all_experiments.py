import argparse
from constants import TASK_NAMES, SELECTIVE_ANNOTATION_METHODS


def main(model_name):
    with open("run_all_experiments.sh", "w") as file:
        file.write("#!/bin/bash\n\n")
        for task in TASK_NAMES:
            for method in SELECTIVE_ANNOTATION_METHODS:
                command = f"python main.py --task_name {task} --selective_annotation_method {method} "
                command += f"--model_cache_dir models --data_cache_dir datasets "
                command += (
                    f"--output_dir outputs/{task}/{method} --model_name={model_name}\n"
                )
                file.write(command)

            file.write("\n")

        file.write("\n")
        file.write("python exelify_results.py\n")


if __name__ == "__main__":
    # python generate_run_all_experiments.py --model_name=EleutherAI/gpt-neo-125m
    # python generate_run_all_experiments.py --model_name=EleutherAI/gpt-j-6B
    parser = argparse.ArgumentParser(
        description="Generate run_all_experiments.sh script."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use for all experiments.",
    )
    args = parser.parse_args()

    main(args.model_name)
