import os
import gdown
import subprocess

outfname = "cit_first_floor.zip"
output = f"bosdyn/{outfname}"
if not os.path.exists(output):
    print("Downloading CIT First Floor GraphNav map")
    foref_models_url = "https://drive.google.com/uc?id=1jO4tLMtC56Wi1M5xXJlGEmRRX6f-UW-I"
    gdown.download(foref_models_url, output, quiet=False)
    cmd=f'''
cd bosdyn
unzip {outfname}
rm {outfname}
cd ..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
