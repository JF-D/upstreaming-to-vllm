import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--profile-dir", type=str, default=None)
parser.add_argument("--log", type=str)
args = parser.parse_args()

BASE_DIR = "_cache_profile"
os.system(f"mkdir -p {BASE_DIR}")

profile_dir = args.profile_dir
logfile = args.log
if profile_dir is None:
    profile_dir = logfile.split("/")[-1].split(".")[0]

path = None
with open(f"{logfile}", "r") as fp:
    for line in fp.readlines():
        if "BIRLinker cwd" in line:
            path = line.split()[-1]
neff_name = path.split("/")[-1]

os.system(f"mkdir -p {BASE_DIR}/{profile_dir}")
os.system(f"rm -rf {BASE_DIR}/{profile_dir}/*") # clear

os.system(f"cp {path}/{neff_name}.pb.neff  {BASE_DIR}/{profile_dir}/file.neff")
os.system(f"cd {BASE_DIR}/{profile_dir} && neuron-profile capture -r 32 -n file.neff --num-exec 2 --profile-nth-exec 2 -i 0 -s profile.ntff 2>&1 | tee log.log")
if os.path.exists(f"{BASE_DIR}/{profile_dir}/profile_rank_0_exec_2.ntff"):
    os.system(f"mv {BASE_DIR}/{profile_dir}/profile_rank_0_exec_2.ntff {BASE_DIR}/{profile_dir}/profile.ntff")
os.system(f"touch {BASE_DIR}/{profile_dir}/{neff_name}.txt")

cwdpath = os.path.abspath(os.getcwd())
print(f"Write to {cwdpath}/{BASE_DIR}/{profile_dir}")

if os.path.exists(f"{BASE_DIR}/{profile_dir}/profile.ntff"):
    print(f"Upload to S3: aws s3 cp {BASE_DIR}/{profile_dir} s3://kaena-tempdata/jfduan/{profile_dir} --acl bucket-owner-full-control --recursive")
    os.system(f"aws s3 cp {BASE_DIR}/{profile_dir} s3://kaena-tempdata/jfduan/{profile_dir} --acl bucket-owner-full-control --recursive")

#os.system(f"cd {BASE_DIR}/{profile_dir} && neuron-profile view -n model.neff -s profile_rank_0_exec_2.ntff --output-format perfetto")

