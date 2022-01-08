This is the ROS workspace for turtlebot. It is
created as I go through the Turtlebot tutorial:
https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/

Note that there are many outdated tutorials (dating back to 2015).
The above is referenced by [this blog](https://automaticaddison.com/how-to-launch-the-turtlebot3-simulation-with-ros/)
written in 2020.


## Troubleshooting

### Could NOT find PY_em (missing: PY_EM)

```
-- Could NOT find PY_em (missing: PY_EM)
CMake Error at /opt/ros/noetic/share/catkin/cmake/empy.cmake:30 (message):
  Unable to find either executable 'empy' or Python module 'em'...  try
  installing the package 'python3-empy'
```

1. I checked `/opt/ros/noetic/share/catkin/cmake/empy.cmake:30`. The code uses a `find_python_module` function
   that is custom also present in `empy.cmake`. That function finds a python
   module by running a small piece of python code:
   ```cmake
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
      RESULT_VARIABLE _${module}_status
      OUTPUT_VARIABLE _${module}_location
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
   ```
2. Open that file with sudo and add a line to print out `PYTHON_EXECUTABLE`:
   ```
    message(STATUS "HIHIHIHI: ${PYTHON_EXECUTABLE}")
   ```
   I get the output
   ```
   -- HIHIHIHI: /home/kaiyu/repo/robotdev/turtlebot/build/venv/bin/python3
   ```
   This is great - When I activate virtualenv, `catkin_make` is trying to use the python in my virtualenv.

   So what failed is really the python code.

3. I entered more `message` commands in the `empy.cmake` file and got more info:

   ```cmake
    message(STATUS "MSG: python executable: ${PYTHON_EXECUTABLE}")
    message(STATUS "MSG: module: ${module}")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
      RESULT_VARIABLE _${module}_status
      OUTPUT_VARIABLE _${module}_location
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "MSG: execute_process: RESULT_VARIABLE: ${_${module}_status}")
    message(STATUS "MSG: execute_process: OUTPUT_VARIABLE: ${_${module}_location}")
    ...
   ```
  The output:
  ```
  -- MSG: python executable: /home/kaiyu/repo/robotdev/turtlebot/build/venv/bin/python3
  -- MSG: module: em
  -- MSG: execute_process: RESULT_VARIABLE: No such file or directory
  -- MSG: execute_process: OUTPUT_VARIABLE:
  ```
  Very funny. "No such file or directory" is the problem.

4. Now, inside `turtlebot` virtualenv, start a python shell.
   Note that this python shell will have the SAME path to the `python` executable
   as the one used in cmake. You can make sure by doing `import sys; print(sys.executable)`.

5. Now, I try to run the same python command in the `execute_process` function,
   except I replaced `${PYTHON_EXECUTABLE}` by `python`, and `${module}` by `em`.
   I get, however, a correct output:
   ```
   $ python -c "import re, em; print(re.compile ('/__init__.py.*').sub('',em.__file__))"
   /home/kaiyu/repo/robotdev/turtlebot/venv/turtlebot/lib/python3.8/site-packages/em
   ```
