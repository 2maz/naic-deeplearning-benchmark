# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse

from pathlib import Path
import pandas as pd

AVAILABLE_TESTS = { "fp32": {
            "PyTorch_SSD_FP32": ("ssd", "^.*Average images/sec:.*$", -1),
            "PyTorch_resnet50_FP32": ("resnet50", "^.*Summary: train.loss.*$", 11),
            "PyTorch_gnmt_FP32": ("gnmt", "^.*Training:.*$", 4),
            "PyTorch_ncf_FP32": ("ncf", "^.*best_train_throughput.*$", 7),
            "PyTorch_transformerxlbase_FP32": (
                "transformerxlbase",
                "^.*Training throughput:.*$",
                -2,
            ),
            "PyTorch_transformerxllarge_FP32": (
                "transformerxllarge",
                "^.*Training throughput:.*$",
                -2,
            ),
            "PyTorch_tacotron2_FP32": ("tacotron2", "^.*train_items_per_sec :.*$", -2),
            "PyTorch_waveglow_FP32": ("waveglow", "^.*train_items_per_sec :.*$", -2),
            "PyTorch_bert_large_squad_FP32": (
                "bert_large_squad",
                "^.*training_sequences_per_second :.*$",
                -6,
            ),
            "PyTorch_bert_base_squad_FP32": (
                "bert_base_squad",
                "^.*training_sequences_per_second :.*$",
                -6,
            ),
        },
        "fp16": {
        "PyTorch_SSD_AMP": ("ssd", "^.*Average images/sec:.*$", -1),
        "PyTorch_resnet50_AMP": ("resnet50", "^.*Summary: train.loss.*$", 11),
        "PyTorch_gnmt_FP16": ("gnmt", "^.*Training:.*$", 4),
        "PyTorch_ncf_FP16": ("ncf", "^.*best_train_throughput.*$", 7),
        "PyTorch_transformerxlbase_FP16": (
            "transformerxlbase",
            "^.*Training throughput:.*$",
            -2,
        ),
        "PyTorch_transformerxllarge_FP16": (
            "transformerxllarge",
            "^.*Training throughput:.*$",
            -2,
        ),
        "PyTorch_tacotron2_FP16": ("tacotron2", "^.*train_items_per_sec :.*$", -2),
        "PyTorch_waveglow_FP16": ("waveglow", "^.*train_items_per_sec :.*$", -2),
        "PyTorch_bert_large_squad_FP16": (
            "bert_large_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
        "PyTorch_bert_base_squad_FP16": (
            "bert_base_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
    }
}


def find_direct_subfolders(folder_path):
    try:
        subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
        return subfolders
    except Exception as e:
        return str(e)


def update_throughput(
    list_test,
    name,
    system,
    df,
    path_result,
    precision,
    verbose,
    metric_name: str = "throughput"
) -> tuple[pd.DataFrame, list[str]]:
    column_name, key, pos = list_test[name]
    pattern = re.compile(key)

    path = path_result + "/" + system + "/" + name
    count = 0.000000001
    total_throughput = 0.0

    errors = []

    task_name = column_name
    start_time = -1
    end_time = -1
    exit_code = None
    slurm_job_id = -1

    files = list([x for x in Path(path).glob("*.txt")])
    files.sort(key=lambda x: os.path.getmtime(x))

    for filename in files:
        flag = False
        throughput = 0
        process_summary = False

        # Sift through all lines and only keep the last occurrence
        for i, line in enumerate(open(filename)):
            for match in re.finditer(pattern, line):
                if verbose:
                    print(match.group().split(' ')) # for debug
                try:
                    throughput = float(match.group().split(" ")[pos])
                except:
                    pass
            
            if process_summary:
                _, start_time, end_time, exit_code, slurm_job_id = line.split(" ")
                slurm_job_id = int(slurm_job_id.strip())
                process_summary = False
            elif re.match(r"# TASK START END EXIT_CODE SLURM_JOB_ID", line):
                # Marker is set in benchmark_pytorch .sh
                process_summary = True

        if throughput > 0:
            count += 1
            total_throughput += throughput
            flag = True

        if not flag:
            errors.append(f"{system=} {name=}: could not locate {pattern} in {Path(filename).resolve()}")
        total_throughput = int(round(total_throughput / count, 2))

    m = re.match(r"(.*)_([0-9])x(.*)_(.*)$", system)
    try:
        number_of_gpus = int(m.group(2)) 
    except:
        print(f"Failed to identify number of gpus from: {system}")
        raise
    node = m.group(4)

    row = pd.DataFrame([{
        "system": system,
        "number_of_gpus": number_of_gpus,
        "node": node,
        "task_name": task_name,
        "start_time": int(start_time),
        "end_time": int(end_time),
        "exit_code": exit_code,
        "slurm_job_id": slurm_job_id, 
        "metric_name": metric_name,
        "metric_value": total_throughput,
        "precision": precision,
    }])

    df = pd.concat([df, row], ignore_index=True)
    return df, errors


def main():
    parser = argparse.ArgumentParser(description="Gather benchmark results.")

    parser.add_argument(
        "--path", type=str, default="results_v2", help="path that has the results"
    )
    parser.add_argument(
        "-o","--output-path", type=str, default=None, help="target directory where to store the results"
    )
    parser.add_argument(
        "-v","--verbose", action="store_true", help="add verbose/debug output"
    )

    args = parser.parse_args()

    # list_system: all direct sub folders in the results folder
    list_system = find_direct_subfolders(args.path)
    columns = [
         "system", 
         "node",
         "number_of_gpus",
         "task_name",
         "start_time",
         "end_time",
         "exit_code",
         "slurm_job_id",
         "precision",
         "metric_name",
         "metric_value",
    ]

    df = pd.DataFrame(columns=columns)
    all_errors = []
    for system in list_system:
        for precision in ["fp16", "fp32"]:
            tests = AVAILABLE_TESTS[precision]

            for test_name, value in sorted(tests.items()):
                df, errors = update_throughput(
                    tests,
                    test_name,
                    system,
                    df,
                    args.path,
                    precision,
                    args.verbose,
                )
                all_errors.extend(errors)

    output_path = Path()

    if args.output_path is not None:
        output_path = Path(args.output_path)
    
    output_filename = output_path / f"pytorch-train-throughput-v2.csv"
    print(df)
    df.to_csv(output_filename)

    print(f"Compiled results into {output_filename}")
    error_msg = '\n    '.join(all_errors)
    print(f"\nErrors:\n    {error_msg}")




if __name__ == "__main__":
    main()
