Design
------

In contrast to `spot_ros` which almost entirely provides its
functionality through a `SpotWrapper`, this package (and other
`rbd_spot_` packages) make use of small, simple objects, and offers
functionality mainly through module functions.

This is convenient when you only need to access a small subset of Spot
capabilities, and it also makes the code more modular and portable.


Coding conventions
------------------

The following conventions apply to `rbd_spot_*` packages:

 - If the function involves calling (direct or indirect) Spot SDK
   Services, the function name is camelCase (first letter
   lower-case). Such functions should return if possible
   the amount of time taken to get the response. *This is by
   convention*.

 - If the input variable is the response of a protobuf service, the
   variable is suffixed by `_result`; The `result` is singular even if
   the response is a list (google protobuf's list type);

 - If the output (returned object) is a request to a protobuf service,
   teh variable is suffixed by `_request`

 - If the function is ros-related, prefix the function with `ros_`
