
# Important to Know

* [The documentation page for the Python Client](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/readme).

    For every service, Spot SDK provides a "Client" object. Look through
    the files under `bosdyn/client`.

* Read [Boston Dynamics API Protobuf Guidelines](https://dev.bostondynamics.com/docs/protos/style_guide).

  You need to have a basic understanding of how Google Protobuf works. Check out [this official tutorial](https://developers.google.com/protocol-buffers/docs/pythontutorial#why-use-protocol-buffers).


# How to Find Out the Format (Schema) of a gRPC Protobuf in Spot SDK?

[This page lists all the protos that make up Boston Dynamics's Public API](https://dev.bostondynamics.com/protos/bosdyn/api/readme).
For example, you will find `image.proto`, `estop.proto`, etc. **This Reference is SUPER useful.**

Each `.proto` file defines services and corresponding message types.

Note that you can find the corresponding generated Python file as `<proto_name>_pb2.py` under `bosdyn/api`.
For example, `image_pb2.py`.


## Appendix

### Spot SDK Services

List available Spot SDK services "live on the robot".
```
$ python -m bosdyn.client --user user --password $SPOT_USER_PASSWORD $SPOT_IP dir list
name                             type                                                      authority                                   tokens
--------------------------------------------------------------------------------------------------------------------------------------------
arm-surface-contact              bosdyn.api.ArmSurfaceContactService                       arm-surface-contact.spot.robot              user
auth                             bosdyn.api.AuthService                                    auth.spot.robot
auto-return                      bosdyn.api.auto_return.AutoReturnService                  auto-return.spot.robot                      user
choreography                     bosdyn.api.spot.ChoreographyService                       choreographyservice.spot.robot              user
data                             bosdyn.api.DataService                                    data.spot.robot                             user
data-acquisition                 bosdyn.api.DataAcquisitionService                         data-acquisition.spot.robot                 user
data-acquisition-store           bosdyn.api.DataAcquisitionStoreService                    data-acquisition-store.spot.robot           user
data-buffer                      bosdyn.api.DataBufferService                              buffer.spot.robot                           user
data-buffer-private              bosdyn.api.DataBufferService                              bufferprivate.spot.robot                    user
deprecated-auth                  bdRobotApi.AuthService                                    auth.spot.robot
directory                        bosdyn.api.DirectoryService                               api.spot.robot                              user
directory-registration           bosdyn.api.DirectoryRegistrationService                   api.spot.robot                              user
docking                          bosdyn.api.docking.DockingService                         docking.spot.robot                          user
door                             bosdyn.api.spot.DoorService                               door.spot.robot                             user
echo                             bosdyn.api.EchoService                                    echo.spot.robot                             user
estop                            bosdyn.api.EstopService                                   estop.spot.robot                            user
fault                            bosdyn.api.FaultService                                   fault.spot.robot                            user
graph-nav-service                bosdyn.api.graph_nav.GraphNavService                      graph-nav.spot.robot                        user
gripper-camera-param-service     bosdyn.api.internal.GripperCameraParamService             gripper-camera-param-service.spot.robot     user
image                            bosdyn.api.ImageService                                   api.spot.robot                              user
internal.localnav                bosdyn.api.internal.localnav.LocalNavService              localnav.spot.robot                         user
lease                            bosdyn.api.LeaseService                                   api.spot.robot                              user
license                          bosdyn.api.LicenseService                                 api.spot.robot                              user
local-grid-service               bosdyn.api.LocalGridService                               localgrid.spot.robot                        user
log-annotation                   bosdyn.api.LogAnnotationService                           log.spot.robot                              user
manipulation                     bosdyn.api.ManipulationApiService                         manipulation.spot.robot                     user
metrics-logging                  bosdyn.api.metrics_logging.MetricsLoggingRobotService     metricslogging.spot.robot
network-compute-bridge           bosdyn.api.NetworkComputeBridge                           network-compute-bridge.spot.robot           user
old_directory                    bdRobotApi.DirectoryService                               api.spot.robot                              user
payload                          bosdyn.api.PayloadService                                 payload.spot.robot                          user
payload-registration             bosdyn.api.PayloadRegistrationService                     payload-registration.spot.robot
power                            bosdyn.api.PowerService                                   power.spot.robot                            user
public_api                       bdRobotApi.RobotRpc                                       api.spot.robot                              user
public_estop_api                 bdRobotApi.estop.RobotEstopRpc                            estop-api.spot.robot                        user
ray-cast                         bosdyn.api.internal.RayCastService                        ray-cast.spot.robot                         user
recording-service                bosdyn.api.graph_nav.GraphNavRecordingService             recordingv2.spot.robot                      user
robot-command                    bosdyn.api.RobotCommandService                            command.spot.robot                          user
robot-id                         bosdyn.api.RobotIdService                                 id.spot.robot
robot-mission                    bosdyn.api.mission.MissionService                         robot-mission.spot.robot                    user
robot-state                      bosdyn.api.RobotStateService                              state.spot.robot                            user
robot_images                     bdRobotApi.ImageService                                   api.spot.robot                              user
spot-check                       bosdyn.api.spot.SpotCheckService                          check.spot.robot                            user
time-sync                        bosdyn.api.TimeSyncService                                api.spot.robot                              user
world-objects                    bosdyn.api.WorldObjectService                             world-objects.spot.robot                    user
```
