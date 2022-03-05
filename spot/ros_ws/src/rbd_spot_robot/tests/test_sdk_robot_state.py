import rbd_spot
import rbd_spot_robot.state as spot_state

def test():
    conn = rbd_spot.SpotSDKConn(sdk_name="RobotStateClient")
    c = spot_state.create_client(conn)
    print(spot_state.getRobotState(c))

if __name__ == "__main__":
    test()
