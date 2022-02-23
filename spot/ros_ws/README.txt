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

 - If a function requires as input a client, the `client` will be the
   first input to the function. If it requires both a `client` and a
   connection to Spot (i.e. the `conn` object), the client should
   follow `conn`.

 - If the function involves calling (direct or indirect) Spot SDK
   Services, the function name is camelCase (first letter
   lower-case). Such functions will return, besides the output of the
   service, the time taken to get the response. *This is by
   convention*.

 - If the input variable is the response of a protobuf service, the
   variable is suffixed by `_result`; The `result` is singular even if
   the response is a list (google protobuf's list type); this
   convention only applies to inputs and not temporary variables
   within functions.  For variables that are neither a response nor a
   request, do not use these suffixes. Variables with `_result` or
   `_request` suffixes are assumed to be protobuf objects.  If plural,
   then do 'requests'.

 - If the output (returned object) is a request to a protobuf service,
   teh variable is suffixed by `_request`

 - If the function is ros-related, prefix the function with `ros_`
