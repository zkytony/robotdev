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

# UID/GID for the container user
hostuser=$USER
hostuid=$UID
hostgroup=$(id -gn $hostuser)
hostgid=$(id -g $hostuser)
# allows user to supply a custom suffix
custom_tag_suffix=""

nvidia=""
for arg in "$@"
do
    if parse_var_arg $arg; then
        if [[ $var_name = "hostuser" ]]; then
            hostuser=$var_value
        elif [[ $var_name = "tag-suffix" ]]; then
            custom_tag_suffix="-$var_value"
        else
            echo -e "Unrecognized argument variable: ${var_name}"
        fi
    elif is_flag $arg; then
        if [[ $arg = "--nvidia" ]]; then
            nvidia=".nvidia"
        fi
    fi
done

# Build the docker image.  The `--rm` option is for you to more conveniently
# rebuild the image.
cd $PWD/../  # get to the root of the repository
docker build -f Dockerfile.noetic${nvidia}\
       -t robotdev:noetic$custom_tag_suffix\
       --build-arg hostuser=$hostuser\
       --build-arg hostgroup=$hostgroup\
       --build-arg hostuid=$hostuid\
       --build-arg hostgid=$hostgid\
       --rm\
       .
# Explain the options above:
# -t: tag
