import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str)
args = parser.parse_args()

req_tpt, token_tpt = 0, 0
avg_speed, avg_overhead = 0, 0
with open(args.log, "r") as fp:
    for line in fp.readlines():
        if "Request Throughput:" in line:
            req_tpt = float(line.split()[-2])
        if "Token Throughput:" in line:
            token_tpt = float(line.split()[-2])
        if "AVG SPEED:" in line:
            avg_speed = float(line.split()[-6])
            avg_overhead = float(line.split()[-2])

print(f"{req_tpt:.2f}, {token_tpt:.2f}, {avg_speed}, {avg_overhead}")
