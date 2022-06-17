# Spot CORE

Spot CORE payload reference: https://bostondynamics.force.com/SupportCenter/s/article/Spot-CORE-payload-reference

Spot CORE comes as an assembly of (1) the Spot CORE computer and
(2) [Spot GXP (General Expansion Payload)](https://support.bostondynamics.com/s/article/Spot-General-Expansion-Payload-GXP).
It has a Lidar cable, an ethernet cable, and a "225mm ribbon cable."

## Attaching the Spot CORE

We should do "rearward mount" since our Spot has an arm.

The CORE payload reference above contains the step-by-step attachment procedure.

Couple things to note [Kaiyu's]:

- Terminology. See picture below.

   <img src='https://user-images.githubusercontent.com/7720184/174300757-dd6024e8-9c68-433d-a478-86457b91a2d6.png' width="500px">

   Additional terminology: "payload cable" is the black cable with a wide port that has gazillion number of pins.

- The official procedure says "Loosen the set screw, if needed, to fit the nut into the slot so that it lays flat and moves freely" and "Do not tighten the set screws" --> Indeed - I used a thin Allen Key to loosen the set screw (that means, the tip of the set screw comes back up a bit), so that the t-slot nut is flat when placed into the rail.

- Make sure the hex screws are screwed in to the nut. If you can't seem to screw in (i.e. you seem to be able to take out the screw with bare hands), try
  to apply a bit pressure on the top of the CORE (the center metal region). Or adjust the placement of the CORE slightly, so that the hole on the base plate lies flat on the mounting rail.

- If installed correctly, you should not be able to shake the CORE at all with your hand.


## Using Spot CORE
CORE reference:
https://support.bostondynamics.com/s/article/Spot-CORE-payload-reference

CORE Cockpit
https://dev.bostondynamics.com/docs/payload/spot_core_cockpit


SSH into the Spot CORE:
```
ssh -p 20022 spot@$SPOT_IP
```

Access the Spot CORE server via browser: Visit http://{SPOT_IP}:21443
