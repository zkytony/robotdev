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

   The question is why PYTHON_EXECUTABLE is set like that in the first place.


9. I deactivated my virtualenv and ran `catkin_make` again. I get the same
   value for `PYTHON_EXECUTABLE`! This must have been hard-coded somewhere.


10. I wanted to know whether the PYTHON_EXECUTABLE
    is set incorrectly if I work in a different ROS workspace.
    I created a temporary ROS workspace that is empty. I then ran
    `catkin_make` on it and:

    - the same CMake messages I added were printed (so `em` was still a necessary package)
    - I get a different path for PYTHON_EXECUTABLE:
      `/usr/local/bin/python3`

    Why? Not sure.

    Full output:
    ```
    ~/ws/tmp_ws$ catkin_make
    Base path: /home/kaiyu/ws/tmp_ws
    Source space: /home/kaiyu/ws/tmp_ws/src
    Build space: /home/kaiyu/ws/tmp_ws/build
    Devel space: /home/kaiyu/ws/tmp_ws/devel
    Install space: /home/kaiyu/ws/tmp_ws/install
    Creating symlink "/home/kaiyu/ws/tmp_ws/src/CMakeLists.txt" pointing to "/opt/ros/noetic/share/catkin/cma
    ke/toplevel.cmake"
    ####
    #### Running command: "cmake /home/kaiyu/ws/tmp_ws/src -DCATKIN_DEVEL_PREFIX=/home/kaiyu/ws/tmp_ws/devel
    -DCMAKE_INSTALL_PREFIX=/home/kaiyu/ws/tmp_ws/install -G Unix Makefiles" in "/home/kaiyu/ws/tmp_ws/build"
    ####
    -- The C compiler identification is GNU 9.3.0
    -- The CXX compiler identification is GNU 9.3.0
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Using CATKIN_DEVEL_PREFIX: /home/kaiyu/ws/tmp_ws/devel
    -- Using CMAKE_PREFIX_PATH: /opt/ros/noetic
    -- This workspace overlays: /opt/ros/noetic
    -- Found PythonInterp: /usr/local/bin/python3 (found suitable version "3.8.10", minimum required is "3")
    -- Using PYTHON_EXECUTABLE: /usr/local/bin/python3
    -- Using Debian Python package layout
    -- MSG: python executable: /usr/local/bin/python3
    -- MSG: module: em
    -- MSG: execute_process: RESULT_VARIABLE: 0
    -- MSG: execute_process: OUTPUT_VARIABLE: /home/kaiyu/.local/lib/python3.8/site-packages/em
    -- Found PY_em: /home/kaiyu/.local/lib/python3.8/site-packages/em
    ...
    ```

11. I then created another temporary workspace. Directly running `catkin_make`
    in this empty workspace gives the same output.

    Full output:
     ```
     ~/repo/robotdev/tmp$ catkin_make
     Base path: /home/kaiyu/repo/robotdev/tmp
     Source space: /home/kaiyu/repo/robotdev/tmp/src
     Build space: /home/kaiyu/repo/robotdev/tmp/build
     Devel space: /home/kaiyu/repo/robotdev/tmp/devel
     Install space: /home/kaiyu/repo/robotdev/tmp/install
     Creating symlink "/home/kaiyu/repo/robotdev/tmp/src/CMakeLists.txt" pointing to "/opt/ros/noetic/share/catkin/cmake/toplevel.cmake"
     ####
     #### Running command: "cmake /home/kaiyu/repo/robotdev/tmp/src -DCATKIN_DEVEL_PREFIX=/home/kaiyu/repo/robotdev/tmp/devel -DCMAKE_INSTALL_PREFIX=/home/kaiyu/repo/robotdev/tmp/install -G Unix Makefiles" in "/home/kaiyu/repo/robotdev/tmp/build"
     ####
     -- The C compiler identification is GNU 9.3.0
     -- The CXX compiler identification is GNU 9.3.0
     -- Check for working C compiler: /usr/bin/cc
     -- Check for working C compiler: /usr/bin/cc -- works
     -- Detecting C compiler ABI info
     -- Detecting C compiler ABI info - done
     -- Detecting C compile features
     -- Detecting C compile features - done
     -- Check for working CXX compiler: /usr/bin/c++
     -- Check for working CXX compiler: /usr/bin/c++ -- works
     -- Detecting CXX compiler ABI info
     -- Detecting CXX compiler ABI info - done
     -- Detecting CXX compile features
     -- Detecting CXX compile features - done
     -- Using CATKIN_DEVEL_PREFIX: /home/kaiyu/repo/robotdev/tmp/devel
     -- Using CMAKE_PREFIX_PATH: /opt/ros/noetic
     -- This workspace overlays: /opt/ros/noetic
     -- Found PythonInterp: /usr/local/bin/python3 (found suitable version "3.8.10", minimum required is "3")
     -- Using PYTHON_EXECUTABLE: /usr/local/bin/python3
     -- Using Debian Python package layout
     -- MSG: python executable: /usr/local/bin/python3
     -- MSG: module: em
     -- MSG: execute_process: RESULT_VARIABLE: 0
     -- MSG: execute_process: OUTPUT_VARIABLE: /home/kaiyu/.local/lib/python3.8/site-packages/em
     -- Found PY_em: /home/kaiyu/.local/lib/python3.8/site-packages/em
     ...

     ```

12. Here is the strange thing. I then created a virtualenv under `~/repo/robotdev/tmp`,
    in the same structure as turtlebot's virtualenv. Then, I ran `catkin_make`
     again. But, the `MSG: ...` stuff I added were GONE!

     ```
     (tmp) ~/repo/robotdev/tmp$ virtualenv -p python3 venv/tmp
     created virtual environment CPython3.8.10.final.0-64 in 85ms
       creator CPython3Posix(dest=/home/kaiyu/repo/robotdev/tmp/venv/tmp, clear=False, global=False)
       seeder FromAppData(download=False, pip=latest, setuptools=latest, wheel=latest, pkg_resources=latest, via=copy, app_data_dir=/home/kaiyu/.local/share/virtualenv/seed-app-data/v1.0.1.debian.1)
       activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator
     kaiyu@zephyr:~/repo/robotdev/tmp$ ls
     build  devel  src  venv
     kaiyu@zephyr:~/repo/robotdev/tmp$ catkin_make
     Base path: /home/kaiyu/repo/robotdev/tmp
     Source space: /home/kaiyu/repo/robotdev/tmp/src
     Build space: /home/kaiyu/repo/robotdev/tmp/build
     Devel space: /home/kaiyu/repo/robotdev/tmp/devel
     Install space: /home/kaiyu/repo/robotdev/tmp/install
     ####
     #### Running command: "cmake /home/kaiyu/repo/robotdev/tmp/src -DCATKIN_DEVEL_PREFIX=/home/kaiyu/repo/robotdev/tmp/devel -DCMAKE_INSTALL_PREFIX=/home/kaiyu/repo/robotdev/tmp/install -G Unix Makefiles" in "/home/kaiyu/repo/robotdev/tmp/build"
     ####
     -- Using CATKIN_DEVEL_PREFIX: /home/kaiyu/repo/robotdev/tmp/devel
     -- Using CMAKE_PREFIX_PATH: /opt/ros/noetic
     -- This workspace overlays: /opt/ros/noetic
     -- Found PythonInterp: /usr/local/bin/python3 (found suitable version "3.8.10", minimum required is "3")
     -- Using PYTHON_EXECUTABLE: /usr/local/bin/python3
     -- Using Debian Python package layout
     -- Using empy: /home/kaiyu/.local/lib/python3.8/site-packages/em
     ```
    This means the `empy.cmake` file I modified was not executed. WHY?

    OH I SEE. Because it was successful before and that progress was
    somehow recorded in the `build/` directory in the workspace. After
    I remove `build/` and ran `catkin_make` again, I do see my modified
    `empy.cmake` being run again:
      ```
      (tmp) ~/repo/robotdev/tmp$ rrm -rf build
      (tmp) ~/repo/robotdev/tmp$ catkin_make
      Base path: /home/kaiyu/repo/robotdev/tmp
      Source space: /home/kaiyu/repo/robotdev/tmp/src
      Build space: /home/kaiyu/repo/robotdev/tmp/build
      Devel space: /home/kaiyu/repo/robotdev/tmp/devel
      Install space: /home/kaiyu/repo/robotdev/tmp/install
      ####
      #### Running command: "cmake /home/kaiyu/repo/robotdev/tmp/src -DCATKIN_DEVEL_PREFIX=/home/kaiyu/repo/robotdev/tmp/devel -DCMAKE_INSTALL_PREFIX=/home/kaiyu/repo/robotdev/tmp/install -G Unix Makefiles" in "/home/kaiyu/repo/robotdev/tmp/build"
      ####
      -- The C compiler identification is GNU 9.3.0
      -- The CXX compiler identification is GNU 9.3.0
      -- Check for working C compiler: /usr/bin/cc
      -- Check for working C compiler: /usr/bin/cc -- works
      -- Detecting C compiler ABI info
      -- Detecting C compiler ABI info - done
      -- Detecting C compile features
      -- Detecting C compile features - done
      -- Check for working CXX compiler: /usr/bin/c++
      -- Check for working CXX compiler: /usr/bin/c++ -- works
      -- Detecting CXX compiler ABI info
      -- Detecting CXX compiler ABI info - done
      -- Detecting CXX compile features
      -- Detecting CXX compile features - done
      -- Using CATKIN_DEVEL_PREFIX: /home/kaiyu/repo/robotdev/tmp/devel
      -- Using CMAKE_PREFIX_PATH: /opt/ros/noetic
      -- This workspace overlays: /opt/ros/noetic
      -- Found PythonInterp: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3 (found suitable version "3.8.10", minimum required is "3")
      -- Using PYTHON_EXECUTABLE: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3
      -- Using Debian Python package layout
      -- MSG: python executable: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3
      -- MSG: module: em
      -- MSG: execute_process: RESULT_VARIABLE: 1
      -- MSG: execute_process: OUTPUT_VARIABLE:
      -- Could NOT find PY_em (missing: PY_EM)
      ...
      ```
     Observations:
     * Notice that the virtualenv `(tmp)` is activated.
     * Notice that the `PYTHON_EXECUTABLE` in this case points
     to the virtualenv's python binary.
     * Notice also that `RESULT_VARIABLE: 1` is 1 which means there
       was an error - this time the error was not that the
       python path is wrong (showing progress!), but that
       the command `import em` failed. That is reasonable because
       I didn't install em in this new workspace.

     This suggests that using virtualenv's python has nothing wrong in principle.

     Then, I did `pip install em` under the `(tmp)` virtualenv.
     I then ran `catkin_make` again and I don't get the PY_EM
     not found error any more!:
       ```
       (tmp) kaiyu@zephyr:~/repo/robotdev/tmp$ catkin_make
        Base path: /home/kaiyu/repo/robotdev/tmp
        Source space: /home/kaiyu/repo/robotdev/tmp/src
        Build space: /home/kaiyu/repo/robotdev/tmp/build
        Devel space: /home/kaiyu/repo/robotdev/tmp/devel
        Install space: /home/kaiyu/repo/robotdev/tmp/install
        ####
        #### Running command: "cmake /home/kaiyu/repo/robotdev/tmp/src -DCATKIN_DEVEL_PREFIX=/home/kaiyu/repo/rob
        otdev/tmp/devel -DCMAKE_INSTALL_PREFIX=/home/kaiyu/repo/robotdev/tmp/install -G Unix Makefiles" in "/home
        /kaiyu/repo/robotdev/tmp/build"
        ####
        -- Using CATKIN_DEVEL_PREFIX: /home/kaiyu/repo/robotdev/tmp/devel
        -- Using CMAKE_PREFIX_PATH: /opt/ros/noetic
        -- This workspace overlays: /opt/ros/noetic
        -- Using PYTHON_EXECUTABLE: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3
        -- Using Debian Python package layout
        -- MSG: python executable: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3
        -- MSG: module: em
        -- MSG: execute_process: RESULT_VARIABLE: 0
        -- MSG: execute_process: OUTPUT_VARIABLE: /home/kaiyu/repo/robotdev/tmp/venv/tmp/lib/python3.8/site-packa
        ges/em
        -- Found PY_em: /home/kaiyu/repo/robotdev/tmp/venv/tmp/lib/python3.8/site-packages/em
        -- Using empy: /home/kaiyu/repo/robotdev/tmp/venv/tmp/lib/python3.8/site-packages/em

       ```

     All of these points to I should just **clean up the
     build directory of turtlebot's workspace**. That should
     fix (OOOF!)

 This is finally resolved! Lessons:

 - It's ok to use virtualenv
 - clean up build/ (and possibly devel/). This should clear up any pre-existing CMake environment variables.
 - adding message in cmake files is a good way to debug.


### ImportError: "from catkin_pkg.package import parse_package" failed: No module named 'catkin_pkg'

This happens when both building using `/usr/local/bin/python3` as well as
using a virtualenv's python.

As suggested in [this ROS Answers post](https://answers.ros.org/question/337135/catkin_make-no-module-named-catkin_pkg/?answer=337155#post-id-337155),
this error means you have not installed `catkin_pkg`. You can install it with
```
sudo apt install python3-catkin-pkg
```
or install the [PYPI](https://pypi.org/project/catkin-pkg/) version:
```
pip install catkin-pkg
```
I did the latter while activating the `(tmp)` virtualenv (this is still part of
testing if basic `catkin_make ` works). This error is resolved afterwards.
However, I see a new error:

#### Can't find __main__ module in ... `/em`
```
/home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3: can't find '__main__' module in '/home/kaiyu/repo/rob
otdev/tmp/venv/tmp/lib/python3.8/site-packages/em'
```
It's again related to em!

GEEZ! I did
```
pip uninstall em
pip install empy
```
and it worked (inside the virtualenv)!

The full output of a successful `catkin_make` build:
```
(tmp) kaiyu@zephyr:~/repo/robotdev/tmp$ catkin_make
Base path: /home/kaiyu/repo/robotdev/tmp
Source space: /home/kaiyu/repo/robotdev/tmp/src
Build space: /home/kaiyu/repo/robotdev/tmp/build
Devel space: /home/kaiyu/repo/robotdev/tmp/devel
Install space: /home/kaiyu/repo/robotdev/tmp/install
####
#### Running command: "cmake /home/kaiyu/repo/robotdev/tmp/src -DCATKIN_DEVEL_PREFIX=/home/kaiyu/repo/robotdev/tmp/devel -DCMAKE_INSTALL_PREFIX=/home/kaiyu/repo/robotdev/tmp/install -G Unix Makefiles" in "/home/kaiyu/repo/robotdev/tmp/build"
####
-- The C compiler identification is GNU 9.3.0
-- The CXX compiler identification is GNU 9.3.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Using CATKIN_DEVEL_PREFIX: /home/kaiyu/repo/robotdev/tmp/devel
-- Using CMAKE_PREFIX_PATH: /opt/ros/noetic
-- This workspace overlays: /opt/ros/noetic
-- Found PythonInterp: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3 (found suitable version "3.8.10", minimum required is "3")
-- Using PYTHON_EXECUTABLE: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3
-- Using Debian Python package layout
-- MSG: python executable: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3
-- MSG: module: em
-- MSG: execute_process: RESULT_VARIABLE: 0
-- MSG: execute_process: OUTPUT_VARIABLE: /home/kaiyu/repo/robotdev/tmp/venv/tmp/lib/python3.8/site-packages/em.py
-- Found PY_em: /home/kaiyu/repo/robotdev/tmp/venv/tmp/lib/python3.8/site-packages/em.py
-- Using empy: /home/kaiyu/repo/robotdev/tmp/venv/tmp/lib/python3.8/site-packages/em.py
-- Using CATKIN_ENABLE_TESTING: ON
-- Call enable_testing()
-- Using CATKIN_TEST_RESULTS_DIR: /home/kaiyu/repo/robotdev/tmp/build/test_results
-- Forcing gtest/gmock from source, though one was otherwise available.
-- Found gtest sources under '/usr/src/googletest': gtests will be built
-- Found gmock sources under '/usr/src/googletest': gmock will be built
-- Found PythonInterp: /home/kaiyu/repo/robotdev/tmp/venv/tmp/bin/python3 (found version "3.8.10")
-- Found Threads: TRUE
-- Using Python nosetests: /usr/bin/nosetests3
-- catkin 0.8.10
-- BUILD_SHARED_LIBS is on
-- BUILD_SHARED_LIBS is on
-- Configuring done
-- Generating done
-- Build files have been written to: /home/kaiyu/repo/robotdev/tmp/build
####
#### Running command: "make -j12 -l12" in "/home/kaiyu/repo/robotdev/tmp/build"
####
```

Applied what I learned here to build the turtlebot workspace
and I was able to move past the same errors!

### Could not find the required component 'turtlebot3_msgs'

The problem is `turtlebot3_msgs` and `turtlebot3` are not
installed on my computer. These two are required
in order to build `turtlebot_simulations`, according to the
[documentations](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/).
The installation instructions of those two packages can be found [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/).
Basically,
```
```
Note, you should install ROS pacakges with `apt` whenever possible. It
does not interfere with your virtualenv, because the path
to ROS's installation is also part of your PYTHONPATH (because you
have sourced ROS's setup.bash).
```
sudo apt-get install ros-noetic-turtlebot3-msgs
sudo apt-get install ros-noetic-turtlebot3
```

**The build of turtlebot3-simulations was successful after fixing this!!!**

Complete log of successful build running my `setup_turtlebot.bash` script:
```
Force remove on! Using /bin/rm
$ Removing /tmp/setup.sh.q7i2CEzhLq
WARNING: Skipping em as it is not installed.
Requirement already satisfied: empy in ./turtlebot/venv/turtlebot/lib/python3.8/site-packages (3.3.4)
Requirement already satisfied: catkin-pkg in ./turtlebot/venv/turtlebot/lib/python3.8/site-packages (0.4.24)
Requirement already satisfied: python-dateutil in ./turtlebot/venv/turtlebot/lib/python3.8/site-packages (from catkin-pkg) (2.8.2)
Requirement already satisfied: docutils in ./turtlebot/venv/turtlebot/lib/python3.8/site-packages (from catkin-pkg) (0.18.1)
Requirement already satisfied: pyparsing in ./turtlebot/venv/turtlebot/lib/python3.8/site-packages (from catkin-pkg) (3.0.6)
Requirement already satisfied: six>=1.5 in ./turtlebot/venv/turtlebot/lib/python3.8/site-packages (from python-dateutil->catkin-pkg) (1.16.0)
Base path: /home/kaiyu/repo/robotdev/turtlebot
Source space: /home/kaiyu/repo/robotdev/turtlebot/src
Build space: /home/kaiyu/repo/robotdev/turtlebot/build
Devel space: /home/kaiyu/repo/robotdev/turtlebot/devel
Install space: /home/kaiyu/repo/robotdev/turtlebot/install
####
#### Running command: "make cmake_check_build_system" in "/home/kaiyu/repo/robotdev/turtlebot/build"
####
####
#### Running command: "make -j12 -l12" in "/home/kaiyu/repo/robotdev/turtlebot/build"
####
[  0%] Built target sensor_msgs_generate_messages_cpp
[  0%] Built target std_msgs_generate_messages_nodejs
[  0%] Built target nav_msgs_generate_messages_lisp
[  0%] Built target turtlebot3_msgs_generate_messages_nodejs
[  0%] Built target nav_msgs_generate_messages_eus
[  0%] Built target geometry_msgs_generate_messages_eus
[  0%] Built target actionlib_msgs_generate_messages_eus
[  0%] Built target nav_msgs_generate_messages_nodejs
[  0%] Built target roscpp_generate_messages_cpp
[  0%] Built target nav_msgs_generate_messages_cpp
[  0%] Built target rosgraph_msgs_generate_messages_eus
[  0%] Built target geometry_msgs_generate_messages_py
[  0%] Built target sensor_msgs_generate_messages_lisp
[  0%] Built target sensor_msgs_generate_messages_eus
[  0%] Built target rosgraph_msgs_generate_messages_nodejs
[  0%] Built target actionlib_msgs_generate_messages_py
[  0%] Built target sensor_msgs_generate_messages_py
[  0%] Built target roscpp_generate_messages_eus
[  0%] Built target tf2_msgs_generate_messages_eus
[  0%] Built target actionlib_generate_messages_cpp
[  0%] Built target std_msgs_generate_messages_lisp
[  0%] Built target geometry_msgs_generate_messages_lisp
[  0%] Built target roscpp_generate_messages_py
[  0%] Built target tf_generate_messages_cpp
[  0%] Built target tf_generate_messages_eus
[  0%] Built target turtlebot3_msgs_generate_messages_py
[  0%] Built target std_msgs_generate_messages_cpp
[  0%] Built target rosgraph_msgs_generate_messages_lisp
[  0%] Built target geometry_msgs_generate_messages_nodejs
[  0%] Built target rosgraph_msgs_generate_messages_py
[  0%] Built target geometry_msgs_generate_messages_cpp
[  0%] Built target roscpp_generate_messages_lisp
[  0%] Built target tf2_msgs_generate_messages_cpp
[  0%] Built target tf2_msgs_generate_messages_py
[  0%] Built target actionlib_generate_messages_lisp
[  0%] Built target actionlib_generate_messages_eus
[  0%] Built target tf_generate_messages_py
[  0%] Built target rosgraph_msgs_generate_messages_cpp
[  0%] Built target actionlib_msgs_generate_messages_cpp
[  0%] Built target actionlib_msgs_generate_messages_lisp
[  0%] Built target actionlib_generate_messages_py
[  0%] Built target actionlib_msgs_generate_messages_nodejs
[  0%] Built target nav_msgs_generate_messages_py
[  0%] Built target tf_generate_messages_nodejs
[  0%] Built target std_msgs_generate_messages_eus
[  0%] Built target tf_generate_messages_lisp
[  0%] Built target tf2_msgs_generate_messages_nodejs
[  0%] Built target tf2_msgs_generate_messages_lisp
[  0%] Built target actionlib_generate_messages_nodejs
[  0%] Built target turtlebot3_msgs_generate_messages_cpp
[  0%] Built target gazebo_msgs_generate_messages_lisp
[  0%] Built target std_msgs_generate_messages_py
[  0%] Built target turtlebot3_msgs_generate_messages_eus
[  0%] Built target gazebo_msgs_generate_messages_cpp
[  0%] Built target roscpp_generate_messages_nodejs
[  0%] Built target turtlebot3_msgs_generate_messages_lisp
[  0%] Built target sensor_msgs_generate_messages_nodejs
[  0%] Built target dynamic_reconfigure_generate_messages_nodejs
[  0%] Built target trajectory_msgs_generate_messages_lisp
[  0%] Built target std_srvs_generate_messages_py
[  0%] Built target std_srvs_generate_messages_lisp
[  0%] Built target dynamic_reconfigure_generate_messages_lisp
[  0%] Built target dynamic_reconfigure_gencfg
[  0%] Built target dynamic_reconfigure_generate_messages_py
[  0%] Built target dynamic_reconfigure_generate_messages_eus
[  0%] Built target std_srvs_generate_messages_eus
[  0%] Built target gazebo_ros_gencfg
[  0%] Built target gazebo_msgs_generate_messages_eus
[  0%] Built target std_srvs_generate_messages_nodejs
[  0%] Built target trajectory_msgs_generate_messages_cpp
[  0%] Built target trajectory_msgs_generate_messages_eus
[  0%] Built target gazebo_msgs_generate_messages_nodejs
[  0%] Built target trajectory_msgs_generate_messages_py
[  0%] Built target trajectory_msgs_generate_messages_nodejs
[  0%] Built target std_srvs_generate_messages_cpp
[  0%] Built target gazebo_msgs_generate_messages_py
[  0%] Built target dynamic_reconfigure_generate_messages_cpp
[ 50%] Built target turtlebot3_fake_node
[100%] Built target turtlebot3_drive
```

## roslaunch autocompletion cannot find turtlebot3-simulation's packages

I wanted to run the command
```
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```
Although `turtlebot3_gazebo` is present in `turtlebot3-simulation`, autocompletion
does not find it.

I stumbled upon [this answer](https://answers.ros.org/question/171266/roslaunch-cannot-find-package/?answer=171276#post-id-171276)
that mentioned this is because you have a problem with your `ROS_PACKAGE_PATH`.
I then found on [ROS Wiki](http://wiki.ros.org/ROS/EnvironmentVariables) that:
>  the `ROS_ROOT` and `ROS_PACKAGE_PATH` enable ROS to locate packages and stacks in the filesystem. You must also set the PYTHONPATH so that the Python interpreter can find ROS libraries.
Note that by default (even in virtualenv), `PYTHONPATH` is empty (that doesn't mean
Python has nowhere to look for packages. Check out [this post](https://stackoverflow.com/questions/20966157/pythonpath-showing-empty-in-ubuntu-13-04)).
