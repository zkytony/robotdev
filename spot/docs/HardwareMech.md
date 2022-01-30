# Facts about Spot Hardware and Mechanical Design:

1. it has 5 pairs of stereo camears that provide black and white images and video.

2. A "hip" refers to the joint that connects the body and a leg, shown below

   ![hip](https://i.imgur.com/pyMhUtQ.png)

   Each hip joint has two _actuators_. One rotates around the X axis, and one rotates around the Y axis.

   >An [actuator](https://en.wikipedia.org/wiki/Actuator), upon receiving a control signal, converts energy (e.g. electric current, hydraulic pressure, pnumatic pressure) into mechanical motion.

    This diagram clearly defines the axes of Spot. It is quite standard: X is forward, Y is left, and Z is up.

    <img src="https://d33wubrfki0l68.cloudfront.net/dd322f87de0e52e2cf381e96d4392b135b6dca61/8fd3c/_images/spotframes.png" width="500px">

3. A knee joint connects the upper leg and the lower leg. It has one actuator (extension range from 14 to 160 degrees.

4. Legs are referred to as "front" or "hind", and "left" or "right". For example, the front left leg is FL; the hind left leg is HL.

   * fl.hx refers to the front left hip X (actuator('s reading))
   * fl.hy refers to the front left hip Y (actuator('s reading))
   * fl.kn refers to the front left knee (actuator('s reading))
