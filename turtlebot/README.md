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

6. I stumbled upon [this Stackoverflow](https://stackoverflow.com/questions/6797395/cmake-execute-process-always-fails-with-no-such-file-or-directory-when-i-cal)
   and it seems like `execute_process` outputting `No such file or directory`
   is a sign that the arguments passed into `executable_process` is wrong.
   In fact `No such file or directory` means that the executable binary
   itself does not exist (in my case) [[ref](https://unix.stackexchange.com/questions/413642/running-executable-file-no-such-file-or-directory#:~:text=No%20such%20file%20or%20directory%22%20means%20that%20either%20the%20executable,also%20need%20other%20libraries%20themselves.&text=then%20the%20problem%20can%20be,in%20the%20library%20search%20path.)].

7. Then I discovered that if I change the cmake command to:
   ```
   execute_process(python ...)
   ```
   while activated in turtlebot virtualenv, the output of `execute_process` makes sense
   (the `RESULT_VARIABLE` equals to 0 and `OUTPUT_VARIABLE` equals to the path
   to the `em` path under the virtualenv.

    **THIS MAKES NO SENSE TO ME.** It is not about quotation.

8. NO, it actually makes sense. The `${PYTHON_EXECUTABLE}` path, though looking legit, is not valid

    ```
    $ /home/kaiyu/repo/robotdev/turtlebot/build/venv/bin/python3
    bash: /home/kaiyu/repo/robotdev/turtlebot/build/venv/bin/python3: No such file or directory
    ```

    CMake actually took the part after the second colon as the `RESULT_VARIABLE`.

   The correct path should be:
   ```
   /home/kaiyu/repo/robotdev/turtlebot/venv/turtlebot/bin/python3
   ```

   *THIS IS THE PROBLEM. CMake got the PYTHON_EXECUTABLE variable set wrong.*
   I don't actually know why it would come up with `/build/venv/` at all.
