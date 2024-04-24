import argparse
from constants import TASK_NAMES, SELECTIVE_ANNOTATION_METHODS


def generate_commands(model_name, subsample):
    escaped_model_name = model_name.split("/")[-1]
    for task in TASK_NAMES:
        for method in SELECTIVE_ANNOTATION_METHODS:
            command = f"python main.py --task_name {task} --selective_annotation_method {method} "
            command += f"--model_cache_dir models --data_cache_dir datasets "
            if subsample:
                command += f"--output_dir outputs/{escaped_model_name}/{task}_subsampled/{method} --model_name {model_name}  --subsample True"
            else:
                command += f"--output_dir outputs/{escaped_model_name}/{task}_full/{method} --model_name {model_name}"
            yield command


def main(model_name, subsample):
    with open("run_all_experiments.sh", "w") as file:
        file.write("#!/bin/bash\n\n")
        for command in generate_commands(model_name, subsample):
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
    parser.add_argument(
        "--subsample",
        type=bool,
    )
    args = parser.parse_args()

    main(args.model_name, args.subsample)
