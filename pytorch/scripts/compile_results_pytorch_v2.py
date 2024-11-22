# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse

from pathlib import Path

import pandas as pd

list_test_fp32 = {
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
    }

list_test_fp16 = {
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


def find_direct_subfolders(folder_path):
    try:
        subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
        return subfolders
    except Exception as e:
        return str(e)


def gather_throughput(
    list_test, name, system, df, path_result
):
    column_name, key, pos = list_test[name]
    pattern = re.compile(key)
    path = path_result + "/" + system + "/" + name
    count = 0.000000001
    total_throughput = 0.0

    errors = []

    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                flag = False
                throughput = 0
                # Sift through all lines and only keep the last occurrence
                for i, line in enumerate(open(os.path.join(path, filename))):
                    for match in re.finditer(pattern, line):
                        if True:
                            print(match.group().split(' ')) # for debug
                        try:
                            throughput = float(match.group().split(" ")[pos])
                        except:
                            pass

                if throughput > 0:
                    count += 1
                    total_throughput += throughput
                    flag = True

                if not flag:
                    errors.append(f"{system=} {name=}: could not locate {pattern} in {filename}")
        df.at[system, column_name] = int(round(total_throughput / count, 2))
    else:
        df.at[system, column_name] = 0

    return errors


def main():
    parser = argparse.ArgumentParser(description="Gather benchmark results.")

    parser.add_argument(
        "--path", type=str, default="results_v2", help="path that has the results"
    )
    parser.add_argument(
        "-o","--output-path", type=str, default=None, help="target directory where to store the results"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Choose benchmark precision",
    )

    args = parser.parse_args()

    if args.precision == "fp32":
        list_test = list_test_fp32
    elif args.precision == "fp16":
        list_test = list_test_fp16
    else:
        sys.exit(
            "Wrong precision: " + args.precision + ", choose between fp32 and fp16"
        )


    # list_system: all direct sub folders in the results folder
    list_system = find_direct_subfolders(args.path)

    columns = []

    for test_name, value in sorted(list_test.items()):
        columns.append(list_test[test_name][0])

    df_throughput = pd.DataFrame(index=list_system, columns=columns)
    df_throughput = df_throughput.sort_index()

    df_throughput = df_throughput.fillna(-1.0)

    all_errors = []
    for system in list_system:
        for test_name, value in sorted(list_test.items()):
            errors = gather_throughput(
                list_test,
                test_name,
                system,
                df_throughput,
                args.path,
            )
            all_errors.extend(errors)

    output_path = Path()
    if args.output_path is not None:
        output_path = Path(args.output_path)

    output_filename = output_path / f"pytorch-train-throughput-v2-{args.precision}.csv"
    df_throughput.to_csv(output_filename)
    print(f"Compiled results into {output_filename}")
    error_msg = '\n    '.join(all_errors)
    print(f"\nErrors:\n    {error_msg}")




if __name__ == "__main__":
    main()
