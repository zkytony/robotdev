# Lease

Check out [lease_service.proto](https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#lease-service-proto) and [lease.proto](https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#lease-proto).

The important thing to know is:
- AcquireLease ([acquire](https://github.com/boston-dynamics/spot-sdk/blob/2fcd195/python/bosdyn-client/src/bosdyn/client/lease.py#L515)): Acquire a lease to a specific resource if the resource is available.
- TakeLease ([take](https://github.com/boston-dynamics/spot-sdk/blob/2fcd195/python/bosdyn-client/src/bosdyn/client/lease.py#L540)): 	Take a lease for a specific resource even if another client has a lease.
