import os
import gdown
import subprocess

os.makedirs("bosdyn", exist_ok=True)
outfname = "assorted_br_maps.zip"
output = f"bosdyn/{outfname}"
if not os.path.exists(output):
    print("Downloading assorted maps (brown robotics)")
    maps_url = "https://drive.google.com/uc?id=1xELxVXuU31_Xz_zoRLKBNZtHPV_R0i9s"
    gdown.download(maps_url, output, quiet=False)
    cmd=f'''
cd bosdyn
unzip {outfname}
rm {outfname}
cd ..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
