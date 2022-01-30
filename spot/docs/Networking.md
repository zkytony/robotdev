# Networking on Spot

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
which will take you to a log-in page:

<img src="https://i.imgur.com/cdi4C66.png" width="500px">


## Have Spot join external WiFi network

Read the [documentation here on "Connecting Spot to a Shared WiFi Network"](https://support.bostondynamics.com/s/article/Spot-network-setup).



It appears from [this discussion](https://support.bostondynamics.com/s/question/0D54X00006K0GrDSAV/spot-network-connection-via-wifi)
that Spot currently only connects to 2.4GHz WiFi, which matches the [manual](https://www.bostondynamics.com/sites/default/files/inline-files/spot-information-for-use-en.pdf),
which says "Connectivity is WiFi 2.4Ghz b/g/n and Gigabit Ethernet".


## Ethernet
A wired connection should be used for updating robot software, accessing logs, and modifying network settings.
