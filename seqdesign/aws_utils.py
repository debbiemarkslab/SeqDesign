import subprocess
import re
import os

if os.path.exists('/n/groups/marks/software/aws-cli/bin/aws'):
    AWS_BIN = '/n/groups/marks/software/aws-cli/bin/aws'
else:
    AWS_BIN = 'aws'


class AWSUtility:
    def __init__(self, s3_base_path, s3_project):
        """
        :param s3_base_path: S3 URL for target S3 folder, e.g. "s3://my-bucket/my-folder"
        :param s3_project: Project name, used as sub-folder of the base path, e.g. "v3"
        """
        self.s3_base_path = s3_base_path.rstrip('/')
        self.s3_project = s3_project

    @staticmethod
    def run_cmd(cmd):
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

    def s3_cp(self, local_file, s3_file, destination='s3'):
        s3_file = f"{self.s3_base_path}/{self.s3_project}/{s3_file}"
        if destination == 's3':
            print("Copying file to AWS S3.")
            src_file, dest_file = local_file, s3_file
        else:
            print("Copying file from AWS S3.")
            src_file, dest_file = s3_file, local_file
        cmd = ['s3', 'cp', src_file, dest_file]
        code, std_out, std_err = self.run_cmd(cmd)
        if code == 0:
            print("Success.")

    def s3_sync(self, local_folder, s3_folder, destination='s3', args=()):
        local_folder = f"{local_folder.rstrip('/')}/"
        s3_folder = f"{self.s3_base_path}/{self.s3_project}/{s3_folder.rstrip('/')}/"
        if destination == 's3':
            print("Syncing data to AWS S3.")
            src_folder, dest_folder = local_folder, s3_folder
        else:
            print("Syncing data from AWS S3.")
            src_folder, dest_folder = s3_folder, local_folder
        cmd = ['s3', 'sync', src_folder, dest_folder, *args]
        code, std_out, std_err = self.run_cmd(cmd)
        if code == 0:
            print("Success.")

    def s3_get_file_grep(self, s3_folder, dest_folder, search_pattern):
        dest_folder = f"{dest_folder.rstrip('/')}/"
        s3_folder = f"{self.s3_base_path}/{self.s3_project}/{s3_folder.rstrip('/')}/"
        print(f"Finding files in {s3_folder} on AWS S3.")
        cmd = ['s3', 'ls', s3_folder]
        code, std_out, std_err = self.run_cmd(cmd)
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
            code, std_out, std_err = self.run_cmd(cmd)
            if code != 0:
                return False
            print("Success.")
        return True
