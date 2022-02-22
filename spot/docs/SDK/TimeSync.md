# Time Sync

The Spot on-board system has a system clock. Your client computer has a clock.
They are independent. These two clocks may not have the same time.

Therefore, it is important to _synchronize_ the two clocks.
[Clock synchronization](https://en.wikipedia.org/wiki/Clock_synchronization)
is a basic distributed systems concept. You must understand two definitions
([reference](https://cse.buffalo.edu/~stevko/courses/cse486/spring14/lectures/06-time.pdf)):

* **[Clock drift](https://en.wikipedia.org/wiki/Clock_drift)** = _Relative_ difference in clock _update rate_ (frequency) between two processes

* **[Clock skew](https://en.wikipedia.org/wiki/Clock_skew)** = _Relative_ difference in clock _values_ (time) between two processes


Independent clocks will drift apart ([clock drift](https://en.wikipedia.org/wiki/Clock_drift)).
You'd better know that:
>All clocks are subject to drift, causing eventual divergence unless resynchronized.
You'd also better know that:
>Clock skew can be caused by many different things, such as wire-interconnect length, temperature variations, variation in intermediate devices, capacitive coupling, material imperfections, and differences in input capacitance on the clock inputs of devices using the clock.


## Time Sync in Spot

Spot SDK provides a [TimeSyncService](https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#timesyncservice)
(also see [this Time Sync doc](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/time_sync)).
This service helps track the difference between Spot's clock and client's clock,
and returns an estimate of this difference to the client.

From the doc: Timestamps in request protos generally need to be specified **relative to the robot’s system clock**.

**In code.** When you do
```python
robot = sdk.create_robot(self.hostname)
```
where `sdk` was created by `create_standard_sdk`,
you have access to a `robot` object ([ref](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/sdk#bosdyn.client.sdk.Sdk.create_robot)),
which is of type [Robot](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot).

You can obtain the clock skew between client (local) and spot clocks
by:
```python
robot.time_sync.endpoint.clock_skew
```
Then, you can calculate the local time with respect to a timestamp from Spot
by taking into account this clock skew:
```python
clock_skew = robot.time_sync.endpoint.clock_skew

# timestamp may be a google.protobuf.timestamp_pb2.Timestamp
seconds = timestamp.seconds - clock_skew.seconds
nanos = timestamp.nanos - clock_skew.nanos
if rtime.nanos < 0:
    # Take care of negative time decimal.
    # Reason is essentially: (1.-1) seconds is technically 0.9 seconds
    # and 9 = (-1 + 10); 0 = 1 - 1.
    # Note that (0.-1) is an invalid time; it will result in -1.9 with
    # the following arithmetic; We take care of that case next.
    rtime.nanos = rtime.nanos + 1000000000
    rtime.seconds = rtime.seconds - 1

if seconds < 0:
   seconds = 0
   nanos = 0
```
