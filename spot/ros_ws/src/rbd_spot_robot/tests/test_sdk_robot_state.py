from rbd_spot_perception.robot_state import RobotStateClient

def test():
    c = RobotStateClient()
    print(c.get())

if __name__ == "__main__":
    test()
