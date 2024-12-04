import os
import argparse
from termcolor import colored

def main():
    parser = argparse.ArgumentParser(description='Gather benchmark results.')

    parser.add_argument('--path', type=str, default='results/48GB',
                        help='path that has the results of all tests')
    args = parser.parse_args()

    print("Check results folder : {}".format(args.path))

    if os.path.exists(args.path):
        lst = os.listdir(args.path)
        lst.sort()
        print(lst)
        for taskname in lst:
            # Get the txt file inside of the folder 
            if not taskname.endswith(".txt"):
                task_dir = os.path.join(args.path, taskname)
                for filename in os.listdir(task_dir):
                    if filename.endswith(".txt"):
                        with open(os.path.join(task_dir, filename), 'r') as f:
                            last_line = f.readlines()[-2]
                            if "SUCCESS" in last_line:
                                print(colored("{: <35} : {: >10}".format(taskname, "successful"), "green"))
                            else:
                                print(colored("{: <35} : {: >10}".format(taskname, "unsuccessful"), "red"))


if __name__ == "__main__":
    main()
