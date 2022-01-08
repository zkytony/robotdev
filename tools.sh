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
