function confirm()
{
    read -p "$1 [y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
	true && return
    fi
    false
}


# https://unix.stackexchange.com/questions/70615/bash-script-echo-output-in-box
function box_out()
{
    local s=("$@") b w
    for l in "${s[@]}"; do
        ((w<${#l})) && { b="$l"; w="${#l}"; }
    done
    tput setaf 3
    echo " -${b//?/-}-"
    for l in "${s[@]}"; do
        printf '| %s%*s%s |\n' "$(tput setaf 4)" "-$w" "$l" "$(tput setaf 3)"
    done
    echo " -${b//?/-}-"
    tput sgr 0
}

function get_major_version {
    # getting ubuntu version (from my dotfiles)
    # $1: a string of form xx.yy where xx is major version,
    #     and yy is minor version.
    v="$(echo ${1} | sed -e 's/\.[0-9]*//')"
    return $(expr $v + 0)
}

function get_minor_version {
    # $1: a string of form xx.yy where xx is major version,
    #     and yy is minor version.
    v="$(echo ${1} | sed -e 's/[0-9]*\.//')"
    return $(expr $v + 0)
}

function ubuntu_version {
    version="$(lsb_release -r | sed -e 's/[\s\t]*Release:[\s\t]*//')"
    echo "$version"
}

function ubuntu_version_greater_than {
    version=$(ubuntu_version)
    get_major_version $version
    major=$?
    get_minor_version $version
    minor=$?
    get_major_version $1
    given_major=$?
    get_minor_version $1
    given_minor=$?

    (( $major > $given_major )) || { (( $major == $given_major )) && (( $minor > $given_minor )); }
}

function ubuntu_version_less_than {
    version=$(ubuntu_version)
    get_major_version $version
    major=$?
    get_minor_version $version
    minor=$?
    get_major_version $1
    given_major=$?
    get_minor_version $1
    given_minor=$?

    (( $major < $given_major )) || { (( $major == $given_major )) && (( $minor < $given_minor )); }
}

function ubuntu_version_equal {
    if ! ubuntu_version_less_than $1; then
	if ! ubuntu_version_greater_than $1; then
	    true && return
	fi
    fi
    false
}


function useros() {
    if ubuntu_version_equal 20.04; then
        source /opt/ros/noetic/setup.bash
        true && return
    elif ubuntu_version_equal 16.04; then
        source /opt/ros/kinetic/setup.bash
        true && return
    else
        echo -e "No suitable ROS version installed"
        false
    fi
}


function check_exists_and_update_submodule {
    if [ -d $1 ]; then
        git submodule update --init --recursive $1
    fi
}

function update_git_submodules {
    # update submodules (clone necessary stuff)
    if confirm "Update git submodules? NOTE: YOU MAY LOSE PROGRESS IF YOUR COMMIT POINTER IS BEHIND SUBMODULE'S LATEST COMMIT."; then
        git submodule update --init --recursive
    fi
}


function get_rlab_interface {
    (echo -e "import netifaces as ni;" ;
     echo -e "for intf in ni.interfaces():" ;
     echo -e "    addrs = ni.ifaddresses(intf);" ;
     echo -e "    if ni.AF_INET not in addrs:" ;
     echo -e "        continue" ;
     echo -e "    ip = addrs[ni.AF_INET][0]['addr'];" ;
     echo -e "    if ip.startswith('138.16.161'):" ;
     echo -e "        print(intf)" ) | python
}

function get_rlab_ip {
    (echo -e "import netifaces as ni;" ;
     echo -e "for intf in ni.interfaces():" ;
     echo -e "    addrs = ni.ifaddresses(intf);" ;
     echo -e "    if ni.AF_INET not in addrs:" ;
     echo -e "        continue" ;
     echo -e "    ip = addrs[ni.AF_INET][0]['addr'];" ;
     echo -e "    if ip.startswith('138.16.161'):" ;
     echo -e "        print(ip)" ) | python
}


# Returns true if this is the first time
# we build the ROS packages related to a robot
function first_time_build
{
    if [ ! -e "$1/src/.DONE_SETUP" ]; then
        # has not successfully setup
        true && return
    else
        false
    fi
}


function build_ros_ws
{
    if catkin_make; then
        echo "$1 SETUP DONE." >> $1/src/.DONE_SETUP
    else
        rm $1/src/.DONE_SETUP
    fi
}
