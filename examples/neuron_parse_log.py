import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logfile", type=str)
args = parser.parse_args()

with open(args.logfile, "r") as fp:
    speeds = []
    for line in fp.readlines():
        if "TIME FOR ONE ITERATION:" in line:
            speed = float(line.strip().split()[-2])
            speeds.append(speed)
speeds = speeds[2:]
speed = sum(speeds) / len(speeds)
print(f"{args.logfile}, {speed:.2f} ms")