# Spot Life Cycle

Useful reference: https://www.generationrobots.com/media/spot-boston-dynamics/spot-user-guide-r2.0-va.pdf
(The instructions in the appendix are copy-and-pasted from this manual)

Youtube Videos: https://www.youtube.com/playlist?list=PLwQV1NIf0wR7WJcQWjRzpFcM-TbQkpu0r

Our Safety Protocols: https://sites.google.com/a/brown.edu/brown-robotics-wiki/robots/spot?authuser=1

## Spot Serial Number and Wifi Password
Before inserting battery into Spot for the first time, take a picture of the label at inside the Battery Slot on Spot. It contains important information:

![image](https://user-images.githubusercontent.com/7720184/140427396-a0af5074-0ef5-45f7-991b-b3f7f4ed4473.png)

The Wifi password is needed for the tablet controller to connect to Spot (which maintains its own Wifi basestation)


## Start-up

### Preparation

Charge Spot and tablet before use if the batteries are below 50%. It takes 2-3 hours to charge the Spot battery. If Spot is docked, only need to make sure the tablet is charged.

### Start-up from the Box
When you just moved Spot out of the box and it is turned off,
follow the instructions [in the official documentation](https://support.bostondynamics.com/s/article/Startup-Procedure)
and [the official tutorial video](https://www.youtube.com/watch?v=NiEFDtUhYKA&list=PLwQV1NIf0wR7WJcQWjRzpFcM-TbQkpu0r&index=10).

Note that much of the steps for starting Spot from the Dock is the same, in particular, the steps on the controller.

The steps are, roughly:
1. Unbox Spot. Two people should lift the spot. Each person will grab two handles (ideally verticall upwards) to lift the Spot out of the box. In the case when there is no payload or no handle, lift spot out by the feet, as follows:
     <img src="https://user-images.githubusercontent.com/7720184/140426294-71ffe27c-8230-45a7-808f-f0fe24bf6ce2.png" width='400px'>

2. Insert Battery into Spot. This is a 3-person job for our spot, because of the arm in the back. Without the arm, one person can do it. Inspect for cracks before inserting the battery.

    ![image](https://user-images.githubusercontent.com/7720184/140426924-01ae16e0-3dc5-4425-821d-49e18c2f304a.png)

3. To power on Spot, press and hold the power button for two seconds. The power button is located on the back, as in the screenshot below:

    ![image](https://user-images.githubusercontent.com/7720184/140427607-ef345461-ac70-4115-8447-03cf94ca4fea.png)


4. Spot initializes, with **loud fan noise**. Wait for the fan noise to lower down. This proabably takes two to three minutes.

6. Once Spot is quiet again, turn on your controller tablet. If the tablet is asleep, press the power button at the top of the controller.

    ![image](https://user-images.githubusercontent.com/7720184/140430443-dfb38243-ce62-4b90-8fea-dfab56842e7a.png)

7. Open the Spot app. If it asks you to upload data, please do not do it. Just tap on “learn more” then tap anywhere on the screen to exist the prompt

     <img src='https://user-images.githubusercontent.com/7720184/140431642-81830481-ec90-4124-af16-bc440bd6492b.png' width="650px">

8. Connect to Spot's WIFI. Tap the “3 bar” symbol on the top left corner then tap the entry for Spot's WiFI. Once connected, enter the user credentials. The credentials are written on a piece of paper taped to the left side of the desk next to the docking station and [in this google doc](https://docs.google.com/document/d/1Bw8M7-g7vHD6bsaYLfeVnYna9jk_0E7pdnKKdDD2Mg0/edit?usp=sharing). If you do not find it, please ask on the Slack channel #spot.

     ![wifi](https://user-images.githubusercontent.com/7720184/140433462-ea493192-ebef-41aa-965c-d1b81fc8f74a.png)

   **Note:** It is ok to turn on the Tablet before turning on Spot. But you would not be able to connect to Spot's Wifi (obviously).


9. **Disengage Spot Motor Lockout.** After this step, you will be able to control Spot. This step is, however, the most complicated in the start-up procedure.

   Spot's motors have mechanical lockouts. Once the motor lockouts are engaged, software cannot control the motors.
   When Spot is first started up, the lockouts are engaged. You must **disengage** it in order to be able to control
   Spot with your Tablet. Refer to the Appendix for more information on Motor Lockout and the signal for the motor lockout button.
   Importantly, when the **Motor Lockout Button has NO COLOR (default), the motor lockout is ENGAGED**. **When the Motor Lockout Button is RED (after press), the motor lockout is DISENGAGED.**

   1. To proceed, first we begin with the screen on the Tablet after WiFi is connected (left pic). Press the Power Button (right pic) to bring up instructions:

         ![controller_start1](https://user-images.githubusercontent.com/7720184/140435666-38e2c17e-e21b-469f-9744-84acd3c15777.png)

        The instructions contain four steps, which will be the following steps in this guide.

    2. Enable the Cut Motor Power button by pressing the hardware key combination (L-Bumper + R-Bumper + B).  (left pic)
    3. Take control of the robot. Press "Take Control" (right pic)

         ![controller_start2](https://user-images.githubusercontent.com/7720184/140436822-886d65c9-476f-4a45-b3f7-6fb9320ab58d.png)

    4. Disengage software Cut Motor Power button. Press ACQUIRE CUT MOTOR POWER AUTHORITY.  (left most pic)

    5. Disengage motor lockout button. Press the Motor Lockout button on the back of Spot. You should see Red light after pressing.  (middle and right pic)

        ![controller_start3](https://user-images.githubusercontent.com/7720184/140438836-e40350ca-f27d-4baf-8b2d-d1c10108307b.png)

    6. Finally, **turn on motor**. Then, the Tablet should display the front-facing camera on Spot. Now, start-up is done!

          ![controller_start4](https://user-images.githubusercontent.com/7720184/140439356-949732b9-fe60-4309-a847-c766e14be89c.png)



### Start-up from the Docking Station

1. Power on Spot. Press the Power Button for 2 seconds, until you see a blue light around it.

    <img src="https://user-images.githubusercontent.com/7720184/140432057-debf1a5e-1878-47e2-a7b2-b20e8e14da2d.JPEG" width='500px'>

2. As usual, Spot initializes, with **loud fan noise**. Wait for the fan noise to lower down. This proabably takes two to three minutes.

3. Once the noise has lowered down, turn on the Controller Tablet (see instruction above)

4. Connect to Spot's Wifi (see instructions above)

5. **Disengage Motor Lockout.** Follow the same steps as above. After the motors are powered on, you should see the following screen ono your Tablet:

     <img src='https://user-images.githubusercontent.com/7720184/140439543-e4d84ca8-051c-4657-a41b-c72f5abfa52e.JPEG' width='600px'>

6. Press undock. The robot should follow its programmed undocking procedure, to move itself out and away from the Dock. DO NOT INTERRUPT THIS PROCESS.

7. Now, you should see the Tablet display the front-facing camera view, similar to the last step in the instructions above.


## In-Use

#### Movement Control
See Appendix for Controller.

Modes:
- X: Sit
- B: Stand
- A: Walk

Controls (Walk mode):
- Left joystick: forward / backward
- Right joystick: horizontal rotation (yaw)

Controls (Stand mode):
- Left joystick:  tilt forward / backward
- Right joystick: tilt left / right

#### Pick-and-Place
TODO

#### Door Opening
TODO

#### Arm Control
TODO


## Shutdown

There are two "types" of shutdown: Either (1) you first dock the robot then shut it down, or (2) you first make it Sit, then shut it down. The "shut it down" part are the same. To make the robot Sit, refer to the instructions above. We first describe how the Docking procedure works.

### Docking procedure

Read the figure below:

![docking](https://user-images.githubusercontent.com/7720184/140441618-c57f4f41-acbe-4847-81f8-5aaa1762afeb.png)

#### Issue: Spot doesn't dock completely
Jasmin (Yanqi) observed this. This is her email:
>After starting self-docking mode, the robot moves to the docking station sits and immediately stands up, moves forward then backs into the docking station again. >The process repeats 3 times, then the control pad shows a self-docking fail message.
>
>Please see a video of the complete process at this link.

First of all, Max tried recalibrate the joints and cameras. It can be done easily by: click the three bars, click "Utilities" and then you should find an option.
But that didn't help.

Max found a way to force the docking which works around this issue. Basically, first do automatic docking. As the robot is about to lower down its body onto the
vertical bars of the dock, press "X" (for Sit) and hold it. That will force abort the Docking process when Spot has in fact lnserted itself into the vertical bars, and will not go back up again (because the Docking process has terminated).


### Shut-down procedure
Essentially, reverse the steps in  "Disengage Motor Lockout button" in the start-up procedure.
**MAKE SURE THE SPOT IS EITHER AT REST ON THE FLOOR, OR SECURELY DOCKED.**

1. Turn off Motor Power. Either, in the Tablet Controller, press the Power Icon and then press "OFF" next to "Motor Power". Or, in the Tablet Controller, press CUT POWER on the top-right of the screen. This will turn off the motor power (from the software side).

   <img src='https://user-images.githubusercontent.com/7720184/140443191-51738c07-42b3-4c3c-a5a7-b0065a734a37.png' width='500px'>


2. Engage Motor Lockout by pressing the Motor Lockout button at the back of the robot. The red light should disappear. You should see the LED lights in the front of the robot turning blue.

   <img src='https://user-images.githubusercontent.com/7720184/140443340-db704d1a-085f-434c-8390-94507b5c8117.png' width='500px'>


3. In the Tablet Controller, press "RELINQUISH CUT MOTOR POWER AUTHORITY"
5. Press the "three bars" on the top left, then press "Disconnect." You should see three options: SIGN OUT, REBOOT, SHUTDOWN.
6. Select SHUTDOWN. Select YES for the confirmation.

**Note:** Shutting down the robot REQUIRES CONTROL (that is your Tablet must be the one controlling the robot). So, do not press "RELEASE CONTROL" when shutting down the robot.



## Charging

Use the charger to charge the battery, like this:

<img src='https://user-images.githubusercontent.com/7720184/140426599-99732dd3-e9f7-4810-a5c2-3ff9c8f09359.png' width='550px'>

[Watch this videp.](https://www.youtube.com/watch?v=vg7Cdjj75IU&ab_channel=BostonDynamicsSupport)

# Appendix 1: Spot Controller
Spot is controlled by an android app running on
a tablet. The Spot System includes an Android
gaming tablet with physical buttons for easy
robot control.

![image](https://user-images.githubusercontent.com/7720184/140428387-1a54aeda-b719-4030-a6fe-634a0d191084.png)

![image](https://user-images.githubusercontent.com/7720184/140430378-efcf4abe-2bf5-4abd-bbfb-6fab1aa88f4b.png)


# Appendix 2: Motor Lockout
The lockout button mechanically disconnects
the motor power and can only be activated by
manually pressing the lockout button. Software
cannot turn on the motors while the lockout is
engaged.
To handle Spot safely, the robot must be placed
in lockout mode. To enable lockout:
1. Turn motor power off with the controller.
2. Push the lockout button.
3. Confirm red light is off

![image](https://user-images.githubusercontent.com/7720184/140428765-74901591-eabd-42b2-86f6-967417bd296b.png)

**CAUTION: Never handle Spot UNLESS motor
lockout is engaged.**

# Appendix 3: Handling Spot

Spot has a handle at each hip joint. Use these to
lift, carry, and roll the robot. When using the handles, always make a fist with each hand.

![image](https://user-images.githubusercontent.com/7720184/140429096-61e92ea0-6f03-4d63-b60b-a23c635488e8.png)

- Don’t attempt to lift or move Spot while
it is standing.
- Only move joints with motors in lockout
mode or while Spot is powered off.
- Be careful to avoid pinch points


# Appendix 4: Power & Motor Lights
Refer to the [official Spot user guide](https://www.generationrobots.com/media/spot-boston-dynamics/spot-user-guide-r2.0-va.pdf).

![image](https://user-images.githubusercontent.com/7720184/174320555-af311825-f0ff-49bc-9c22-1def91510297.png)


# Troubleshooting

## Extreme Motor Vibration

**Experience 1 (02/10/2022 ~9:15AM)**: When I try to undock Spot, I accidentally pressed the "Walk" button (A)
while Spot is docked (I am not sure why I could get into that interface when Spot is docked).
Then, Spot stood up above the dock, and I tried to joystick it
out of the Docker (during which spot front-right leg hit dock pillar).

Then, suddenly there is extreme noise (very unpleasant) from the back of
Spot. The operation was still fine. We then pressed "X" to sit the Spot (during
which time the whole back of Spot is vibrating extremely fast), and then
immediately press the power button to shut down the robot (need to withstand the
vibration).

We then rebooted the robot, and everything seemed to be fine again. We did not do calibration again.
