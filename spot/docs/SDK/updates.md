# Important points 3.1.0

- The `–username` and `–password` command line options are deprecated in the Python
  SDK. (**affects, at least, the examples**)

  The solution BD came up with is to ask you to use
  `bosdyn.client.util.add_base_arguments(parser)` when
  creating CLI and not `add_common_arguments()`.

  The username and password are in fact provided through
  `BOSDYN_CLIENT_USERNAME` and `BOSDYN_CLIENT_PASSWORD`
  environment variables (this is how we are doing it in robotdev anyways!!
  SPOT_IP, SPOT_USER_PASSWORD ...)

  _I am not sure when these environment variables are set though._
  It wasn't clear from their [release notes](https://dev.bostondynamics.com/docs/release_notes)
