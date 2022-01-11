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
hostuser=$USER  # the user inside the container
for arg in "$@"
do
    if parse_var_arg arg; then
        if [[ $var_name = "hostuser" ]]; then
            hostuser=$var_value
        else
            echo -e "Unrecognized argument variable: ${var_name}"
        fi
    fi
done

# Build the docker image.  The `--rm` option is for you to more conveniently
# rebuild the image.
cd $PWD/../  # get to the root of the repository
docker build -f Dockerfile.kinetic\
       -t robotdev:kinetic\
       --build-arg hostuser=$hostuser\
       --rm\
       .
