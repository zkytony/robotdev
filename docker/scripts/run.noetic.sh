# Make sure we are in the right directory.
# the rest of the script assumes we are in robotdev/docker
if [[ $PWD = *robotdev ]]; then
    cd docker
elif [[ ! $PWD = *robotdev/docker ]]; then
    echo -e "You must be in either 'robotdev' or the 'robotdev/docker' directory to run this command."
    return 1
fi

# load tools
. "../tools.sh"

# parse args
gui=false
nvidia=false
# allows user to supply a custom suffix
custom_tag_suffix=""
for arg in "$@"
do
    if parse_var_arg $arg; then
        if [[ $var_name = "tag-suffix" ]]; then
            custom_tag_suffix="-$var_value"
        else
            echo -e "Unrecognized argument variable: ${var_name}"
        fi
    elif is_flag $arg; then
        if [[ $arg = "--gui" ]]; then
            gui=true
        elif [[ $arg = "--nvidia" ]]; then
            nvidia=true
        fi
    fi
done

# Create and start the container. Note that this
# script will create a new container and will not
# restart a stopped one. You should run docker
# commands for that.
cd $PWD/../  # get to the root of the repository
if ! $gui; then
    docker run -it\
           --volume $(pwd):/home/$USER/repo/robotdev/\
           --env "TERM=xterm-256color"\
           --privileged\
           --network=host\
           robotdev:noetic$custom_tag_suffix
else
    # Want to support running GUI applications in Docker.
    # Need to forward X11.
    # reference: https://answers.ros.org/question/300113/docker-how-to-use-rviz-and-gazebo-from-a-container/
    XAUTH=/tmp/.docker.xauth
    echo "Preparing Xauthority data..."
    xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
    if [ ! -f $XAUTH ]; then
        if [ ! -z "$xauth_list" ]; then
            echo $xauth_list | xauth -f $XAUTH nmerge -
        else
            touch $XAUTH
        fi
        chmod a+r $XAUTH
    fi

    echo "Done."
    echo ""
    echo "Verifying file contents:"
    file $XAUTH
    echo "--> It should say \"X11 Xauthority data\"."
    echo ""
    echo "Permissions:"
    ls -FAlh $XAUTH
    echo ""
    echo "Running docker..."

    runtime_nvidia=""
    if $nvidia; then
        runtime_nvidia="--runtime=nvidia"
    fi

    docker run -it\
           --volume $(pwd):/home/$USER/repo/robotdev/\
           --env "TERM=xterm-256color"\
           --env "DISPLAY=$DISPLAY"\
           --volume /tmp/.X11-unix/:/tmp/.X11-unix:rw\
           --env "XAUTHORITY=$XAUTH"\
           --volume $XAUTH:$XAUTH\
           --privileged\
           --network=host\
           ${runtime_nvidia}\
           robotdev:noetic$custom_tag_suffix
fi
