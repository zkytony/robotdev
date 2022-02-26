#### Docker utilities ####
function dls
{
    # Lists active and stopped containers
    color_echo "$LBLUE" "Active docker containers:"
    docker container ls
    echo -e ""
    color_echo "$LYELLOW" "Stopped docker containers:"
    docker ps --filter "status=exited"
}

function dsh
{
    if [ "$#" -ne 1 ]; then
        echo -e "Usage: dsh <container>"
    else
        # Starts a session to a container
        if ! drunning $1; then
            docker restart $1
        fi
        docker exec -it $1 bash
    fi
}

function drunning
{
    if [ "$( docker container inspect -f '{{.State.Status}}' $1 )" == "running" ]; then
        true && return
    else
        false
    fi
}

function dils
{
    # List images
    color_echo "$LGRAY" "List of docker images:"
    docker images
}

function dkill
{
    # stop a container
    if [ "$#" -ne 1 ]; then
        echo -e "Usage: dkill <container>"
    else
        if drunning $1; then
            if ask_confirm_yes "Stop container $1? "; then
                docker stop $1
            fi
        else
            echo -e "Container $1 is not running."
        fi
    fi
}
alias dstop="dkill"

function drm
{
    # remove a container
    if [ "$#" -ne 1 ]; then
        echo -e "Usage: drm <container>"
    else
        if ask_confirm_yes "Remove container $1? "; then
            docker rm $1
        fi
    fi
}


function dcommit
{
    if [ "$#" -ne 2 ]; then
        echo -e "Usage: dcommit <container> <image_name>"
    else
        if ! drunning $1; then
            docker commit $1 $2
        else
            echo -e "Unable to commit. The container $1 is still running. Stop it first."
        fi
    fi
}
