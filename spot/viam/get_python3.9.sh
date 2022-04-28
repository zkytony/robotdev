## Download and install Python
PYTHON_VERSION="3.9.12"
if [ ! -d "$HOME/software/Python-$PYTHON_VERSION" ]
then
    mkdir ~/software
    cd ~/software/
    echo -e "Installing Python ${PYTHON_VERSION}"
    if [ ! -f "Python-${PYTHON_VERSION}.tgz" ]
    then
        wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
    fi
    tar xzvf "Python-${PYTHON_VERSION}.tgz"
    cd "Python-${PYTHON_VERSION}"
    ./configure
    make
    make test
fi
PYTHON_39="$HOME/software/Python-$PYTHON_VERSION/python"
sudo ln -sf $PYTHON_39 /usr/bin/python3.9
