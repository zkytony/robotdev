# Movo Bootup Buide
This comes from the Wiki page written by Nishanth and my previous wiki on the [movo_object_search](git@github.com:zkytony/movo_object_search.git) package.

Steps to boot up the Movo and begin using it:

1. Click the power button located at the back of the robot near the red E-Stop button. It's a small metallic button with an LED on its right.

2. Wait for a few minutes. The LED next to the robot should begin alternating between orange and green. If this happens, you're good. If it blinks red rapidly and then turns off, it means the Movo doesn't have enough charge. Plug it in, wait for a while, and try again. If it blinks only orange, check if the E-Stop button is pressed. If so, release the button.

3. Now that the LED is blinking orange and green, it's time to attempt to connect to the robot. By default, Movo uses RLAB wireless with a WiFi router that is mounted on top of the robot [these steps assume Movo is running on wireless]. However, wired and other network configurations are also possible. If you have trouble with the following steps connecting and SSH'ing the robot, please read https://github.com/h2r/movo/blob/master/MovoNetworking.md

4. Ensure that you are on the same network as Movo. RLAB has both a wireless and wired address region, and if Movo is on wireless but your computer is on wired, you'll never be able to ping it. You can verify this by running the command 'ifconfig' in a terminal. It should return an IP address that looks like '138.16.161.xx' or '138.16.160.xx'. If your computer shows the former case (161), then you're on RLAB wireless. Otherwise, you're on RLAB wired and you should try to put your computer on wireless to talk to Movo.

5. Try to run the following command 'ping 138.16.161.17'. This IP address is Movo's default wireless IP address on RLAB wireless. If you are able to ping this, it means Movo is up and running and you're good to continue! If not, try unplugging and then replugging the wireless router on top of Movo to reset it. If you still can't connect, then hook up a monitor and mouse to the ports on Movo's back and access the MOVO2 NUC computer to see if it's connected to the correct network (RLAB) over wireless. Once you put Movo on RLAB, you should be able to ping it.

6. Once you can ping movo, run the following command in terminal with relevant arguments: `sudo route add -net 10.66.171.0 netmask 255.255.255.0 gw 138.16.161.17 dev [network interface]`. The network interface can be obtained from running ifconfig. It will be the word in the left-most column next to your computer's 138.16.xxx.xx IP address and will usually be something like enp3s0 or eth0 or something like that. This command essentially tells your computer to forward all packets going to 10.66.171.* (what the Movo computers think their IP addresses should be) to the Movo's actual IP address on RLAB.

7. Once you have run this command, do the following

    1. Run 'ssh movo@movo2'

    2. Enter 'Welcome00' when prompted for a password

    3. 'ssh movo1', then 'sudo ifconfig wlan0 down', then 'exit'. (without this, movo doesn't publish tf!)

    4. Once inside movo2, clear the space in front of the robot and run the command 'roslaunch movo_bringup movo_system.launch'

    5. Wait for a while. The following things should happen in sequence

        - The Movo should open its fingers

        - The Movo should then close its fingers

        - Almost immediately, the Movo will start moving up and will move its hands to a weird position. Ensure that it doesn't hit anything because that could blow a fuse.

        - The Movo will then start moving back down and move its hands to the tucked position

8. If all is well, the Movo is now in the tucked position. Try to run 'rostopic list'. If that works and you see all of Movo's topics, you're almost done!

9. Check that the ROS_IP and ROS_MASTER_URI environment variables are set correctly. If you do 'echo $ROS_IP' in a terminal, this should print out an IP address that is the same as the one you get when you run ifconfig on YOUR computer. If this is not the case, you need to change it so that it is the same by using the 'export' command in terminal or by finding the place where the ROS_IP is set and changing it there. The ROS_MASTER_URI should be http://movo2:11311/

10. You're good to start working with the Movo! Once you're done, please turn off the robot, make sure the arms don't fall to the ground (tuck them in or something), and plug it in to charge before you leave!


## Shutdown
To shutdown Movo, just press the power button. A red light should flash. Wait until the flash light stops and the noise quiets.
