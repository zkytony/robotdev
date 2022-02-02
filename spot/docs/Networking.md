# Networking on Spot

**Useful references**:

 - [SPOT SYSTEM ADMINISTRATION (PDF)](https://www.bostondynamics.com/sites/default/files/inline-files/spot-system-administration.pdf)

 - [Spot Network Setup](https://support.bostondynamics.com/s/article/Spot-network-setup)


The primary networking capability when using Spot by default WiFi.
One could either join Spot's WiFi access point. Or, Spot can be
a client to an external WiFi network.

Spot API mainly use the **gRPC** (google Remote Procedural Call)
protocol; A gRPC is a specification of a service interface
and the RPCs supported by the service. The input to a service
and the output of a service are both in the format of [Protocol
Buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) (protobuf) messages.
It is a format that looks really like JSON, but not for human-readability
but for data transmission. Read [this](https://grpc.io/docs/what-is-grpc/introduction/)
to learn more about gRPC and protobuf.

The advantage of gRPC is that it is cross-platform and cross-language.
This is illustrated by the figure below:

<img src="https://i.imgur.com/zMr8Ch6.png" width="500px">

Spot has a [detailed documentation](https://dev.bostondynamics.com/docs/concepts/networking) for how to use gRPC and Protobuf
with Spot.


## Some common interfaces

1. The log-in page:

    <img src="https://i.imgur.com/cdi4C66.png" width="400px">

2. The home-page for user:

    <img src="https://i.imgur.com/sTww4U4.png" width="400px">

3. The home-page for admin. It is the same as user's except for the username
   and there are a couple additional capabilities (User Management and Software Update).

    <img src="https://i.imgur.com/YrFRVLL.png" width="400px">




## Ethernet
The detailed steps are documented in [the Ethernet section of this page](https://support.bostondynamics.com/s/article/Spot-network-setup).

A wired connection should be used for updating robot software, accessing logs,
and modifying network settings.

1. Connect spot with your computer using an ethernet.

   <img src="https://i.imgur.com/gy2zndh.jpg" width="300px">

2. Add a manual wired connection (see [this guide](https://linuxconfig.org/how-to-configure-static-ip-address-on-ubuntu-18-10-cosmic-cuttlefish-linux#:~:text=Ubuntu%20Desktop,-The%20simplest%20approach&text=Click%20on%20the%20top%20right,netmask%2C%20gateway%20and%20DNS%20settings.))
   with an IP range 10.0.0.X (netmask 255.255.0.0).

3. Once added, visit [https://10.0.0.3/](https://10.0.0.3/), the default IP for Spot. You will see a log in page. Enter default username and password.


## Connect Spot to external WiFi network (e.g. RLAB)

Recently, Spot has supported 5GHz WiFi (see [this discussion](https://support.bostondynamics.com/s/question/0D54X00006K0GrDSAV/spot-network-connection-via-wifi)).

The steps are in the [documentation here on "Connecting Spot to a Shared WiFi Network"](https://support.bostondynamics.com/s/article/Spot-network-setup).
Basically:

1. Connect Spot with your PC via ethernet.

2. Log in as admin.

3. Go to "Network Setup"->WiFi.

4. Select "client" for WiFi Network Type. (By default, it is "Access Point")

5. Enter the following:

    - Network Name: RLAB
    - Password: (RLAB Password)
    - Frequency Band: All
    - Enable DHCP: FALSE
    - IPv4 address: Enter the static IP you want Spot to have on the RLAB network.
      It must not already be used. For Spot XXXX012, I set it to [138.16.161.12](https://138.16.161.12).
    - Route Prefix: 24
    - Stored Default Route: 0.0.0.0

    Then click "APPLY." If successful, you should see a green message at the top, like this:

    <img src="https://i.imgur.com/wpFr8OF.png" width="350px">

    (Note that the MAC address should be detected automatically after you hit APPLY)

6. On the controller:

    1. On the Select Network page (the page where you see Spot's wifi), join
       the RLAB WiFi network (Connect to a Network).  You need to enter RLAB password.

    2. Once connected, you need to tell the controller which IP address Spot uses.
       That is the static IP address you configured in step 5. You enter that by
       pressing "Add a Robot." Type in the IP address (138.16.161.12) in the interface below:

       <img src="https://i.imgur.com/vXiuaEr.jpg" width="350px">

    3. Once added, you should be able to see an entry for Spot on the right column.
       Press it to connect. The interface changes and you should now enter the user password
       as shown in the following:

       <img src="https://i.imgur.com/LAZbzny.jpg" width="350px">

    4. Press the Green Checkmark once you entered the user password. If success, you
       should be able to control Spot - the rest is the same as normal Spot startup (see [these notes](./LifeCycle.md));
       The controller interface should be the picture on the left, and once Spot
       motor starts, you should reach an interface as on the right (if Spot is docked).

       <img src="https://i.imgur.com/6rSjZYv.jpg" width="250px">
       <img src="https://i.imgur.com/CKRBq6j.jpg" width="250px">



**Note regarding 2.4GHz WiFi.**
It appears from [this discussion](https://support.bostondynamics.com/s/question/0D54X00006K0GrDSAV/spot-network-connection-via-wifi)
that Spot currently only connects to 2.4GHz WiFi, which matches the [manual](https://www.bostondynamics.com/sites/default/files/inline-files/spot-information-for-use-en.pdf),
which says "Connectivity is WiFi 2.4Ghz b/g/n and Gigabit Ethernet".


### How to revert back to using Spot's own broadcasted WiFi network again

1. Before you make any changes in the admin page in browser,
   make sure that:

   - Spot is in a secure position (e.g. docked)

   - Spot motor is shutdown. Mechanical motor lock is engaged (red LED button does not light).

   - You are connected to Spot using an ethernet cable. The ethernet connection works.

2. Do the following through the ethernet connection (i.e. visit https://10.0.0.3 )

   1. Go to Network Setup; Go to WiFi tab.

   2. Select "Access Point" as WiFi Network Type. Press "APPLY". Once successful, you
      should see a green notification at the top:

      <img src="https://i.imgur.com/8FvTx8V.png" width="400px">

   3. On the controller, if the controller was NOT previously connected to Spot,
      then you can simply join Spot's broadcasted network in the Sign-in page
      (aka. Select Network page).

      If the controller was previously connected, you will notice network error
      (red WiFi icon):

       <img src="https://i.imgur.com/uiSF5xl.jpg" width="300px">

      You should:

      1. press the three bars ≡ on the top left, select Disconnect.

      2. Press "Sign Out"

         <img src="https://i.imgur.com/07zSDNf.jpg" width="300px">

      3. Then, in the Sign-in page (aka. Select Network page), select
         the Spot-broadcasted network `spot-BD-XXXXX`). Sign in,
         and the rest should be the same as you enter the familiar
         interface.

         <img src="https://i.imgur.com/7d9M1m8.png" width="250px">
         <img src="https://i.imgur.com/6rSjZYv.jpg" width="250px">








## Join Spot's WiFi network from your computer

Well, typical computers have only one wireless network interface.
That means, if you connect to Spot, you will relinquish your usual
WiFi. Or, you can either plug in ethernet to your computer and
connect to Spot's wifi; Or you could buy an external wifi car.

If you want to connect to Spot's WiFi,
select "spot-BD-XXXXX" on the list of available WiFis;
Note that XXXXX should be specific to the Spot you want
to connect to. Then enter the WiFi Password.

You can then visit [https://192.168.80.3](https://192.168.80.3)
which will take you to a log-in page.
