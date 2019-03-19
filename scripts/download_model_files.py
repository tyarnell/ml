import os
import sys
import argparse

import glob
import pickle
import yaml
import pandas as pd

import subprocess


def main(args):
    command = ['gsutil', '-m','cp', '-r', args.src_url, args.dest_url]
    completed = subprocess.run(command)
    print('returncode:', completed.returncode)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--src-url', required=True)
    parser.add_argument('--dest-url', required=True)
    parser.add_argument('--print-results', default=False)
    args = parser.parse_args()

    main(args)