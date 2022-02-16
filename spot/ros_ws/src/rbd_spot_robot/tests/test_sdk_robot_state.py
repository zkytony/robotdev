from rbd_spot_perception.robot_state import RobotStateClient
import time
from tqdm import tqdm

def test():
    c = RobotStateClient()
    start_time = time.time()
    total_num = 100
    for i in tqdm(range(total_num)):
        c.get()
    time_used = time.time() - start_time
    print(f"Frequency (tested with {total_num} calls): {total_num / time_used}")

if __name__ == "__main__":
    test()
    # output I got:
    # 100%|████████████████████████████████| 100/100 [00:01<00:00, 68.87it/s]
    # Frequency (tested with 100 calls): 68.01107167237708
