#!/usr/bin/env bash
# Follow https://docs.conda.io/en/latest/miniconda.html to install miniconda
# Call this using "source .initialize_conda_env.sh"
version="3.11"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BYSTRO_PYTHON_PATH="${SCRIPT_DIR}/python"
ENV_NAME='bystro'

echo "Script directory: $SCRIPT_DIR, python path $BYSTRO_PYTHON_PATH";

source $(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

if { conda env list | grep ${ENV_NAME}; } >/dev/null 2>&1; then
    echo -e "\n==== ${ENV_NAME} environment exists, activating and ensuring up to date with Python $version  ====\n"
    conda activate ${ENV_NAME}
    conda install python=$version -y -q
else
    echo -e "\n====                Creating ${ENV_NAME} environment with Python $version                  ====\n"
    conda create --name ${ENV_NAME} python=$version -y -q
    conda activate ${ENV_NAME}
fi

echo -e "\n====                         Installing Python development requirements                 ====\n"
find . -name 'requirements-dev.txt' -exec pip install -q -r {} \;
echo -e "\n====                         Setting up development environment                             ====\n"
(cd ${BYSTRO_PYTHON_PATH} && maturin develop -q)
echo -e "\n====                                       Done                                         ====\n"
