import subprocess
import re
import os
from seqdesign.version import VERSION

S3_FOLDER_URL = "s3://markslab-private/autoregressive"

if os.path.exists('/n/groups/marks/software/aws-cli/bin/aws'):
    AWS_BIN = '/n/groups/marks/software/aws-cli/bin/aws'
else:
    AWS_BIN = 'aws'


def run_aws_cmd(cmd):
    try:
        if cmd[0] not in ('aws', AWS_BIN):
            cmd = [AWS_BIN] + cmd
        else:
            cmd[0] = AWS_BIN
        pipes = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
        std_out, std_err = pipes.communicate()
        if pipes.returncode != 0:
            print(f"AWS CLI error: {pipes.returncode}")
            print(std_err.strip())
            return pipes.returncode, None, None
        else:
            return 0, std_out, std_err
    except OSError:
        print("AWS CLI not found.")
        return 1, None, None


def aws_s3_sync(local_folder, s3_folder, destination='s3', version=VERSION, args=()):
    local_folder = local_folder + ('' if local_folder.endswith('/') else '/')
    s3_folder = s3_folder + ('' if s3_folder.endswith('/') else '/')
    s3_folder = f"{S3_FOLDER_URL}/{version}/{s3_folder}"
    if destination == 's3':
        print("Syncing data to AWS S3.")
        src_folder, dest_folder = local_folder, s3_folder
    else:
        print("Syncing data from AWS S3.")
        src_folder, dest_folder = s3_folder, local_folder
    cmd = ['s3', 'sync', src_folder, dest_folder] + list(args)
    code, std_out, std_err = run_aws_cmd(cmd)
    if code == 0:
        print("Success.")


def aws_s3_get_file_grep(s3_folder, dest_folder, search_pattern, version=VERSION):
    s3_folder = s3_folder + ('' if s3_folder.endswith('/') else '/')
    dest_folder = dest_folder + ('' if dest_folder.endswith('/') else '/')
    s3_folder = f"{S3_FOLDER_URL}/{version}/{s3_folder}"
    print(f"Finding files in {s3_folder} on AWS S3.")
    cmd = ['s3', 'ls', s3_folder]
    code, std_out, std_err = run_aws_cmd(cmd)
    if code != 0:
        return False
    filenames = re.findall(search_pattern, std_out)
    if not filenames:
        print("No files found.")
        return False
    print(f"Found: {filenames}")
    for filename in filenames:
        filename = f"{s3_folder}{filename}"
        print(f"Copying file {filename} from AWS S3.")
        cmd = ['s3', 'cp', filename, dest_folder]
        code, std_out, std_err = run_aws_cmd(cmd)
        if code != 0:
            return False
        print("Success.")
    return True


def aws_s3_cp(local_file, s3_file, destination='s3', version=VERSION):
    s3_file = f"{S3_FOLDER_URL}/{version}/{s3_file}"
    if destination == 's3':
        print("Syncing data to AWS S3.")
        src_file, dest_file = local_file, s3_file
    else:
        print("Syncing data from AWS S3.")
        src_file, dest_file = s3_file, local_file
    cmd = ['s3', 'cp', src_file, dest_file]
    code, std_out, std_err = run_aws_cmd(cmd)
    if code == 0:
        print("Success.")
