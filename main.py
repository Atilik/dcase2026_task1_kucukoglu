import subprocess
import sys
import os
import argparse


def run(script, config_file):
    env = os.environ.copy()
    env['CONFIG_FILE'] = config_file
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config file to use (e.g. config_beats.yaml)')
    args = parser.parse_args()

    config_file = args.config
    print(f"Using config: {config_file}")

    pipeline = [
        ("Dataset", "build_dataset.py"),
        ("Training", "train_test.py"),
        ("Results", "summarize_results.py"),
    ]

    for name, script in pipeline:
        print(f"\n------- Process: {name} -------")
        print(f"\nRunning {script}...\n")
        run(script, config_file)
        print(f"\nFinished {script}")

    print("Processes completed. Check the output folders for results.")
