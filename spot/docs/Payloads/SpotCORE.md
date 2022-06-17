# Spot CORE

Spot CORE payload reference: https://bostondynamics.force.com/SupportCenter/s/article/Spot-CORE-payload-reference

Spot CORE comes as an assembly of (1) the Spot CORE computer and
(2) [Spot GXP (General Expansion Payload)](https://support.bostondynamics.com/s/article/Spot-General-Expansion-Payload-GXP).
It has a Lidar cable, an ethernet cable, and a "225mm ribbon cable."

## Attaching the Spot CORE

We should do "rearward mount" since our Spot has an arm.

The CORE payload reference above contains the step-by-step attachment procedure.
Below is the documentation of Kaiyu's experience:




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
