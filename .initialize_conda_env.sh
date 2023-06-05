#!/usr/bin/env bash
# Follow https://docs.conda.io/en/latest/miniconda.html to install miniconda
# Call this using "source .initialize_conda_env.sh"
# Ray 2.4.0 on Mac OS (arm64) does not have stable 3.11 support, see https://docs.ray.io/en/latest/ray-overview/installation.html
version="3.10.11"

source $(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

if { conda env list | grep 'bystro'; } >/dev/null 2>&1; then
    echo -e "\n==== Bystro environment exists, activating and ensuring up to date with Python $version  ====\n"
    conda activate bystro
    conda install python=$version -y -q
else
    echo -e "\n====                Creating Bystro environemnt with Python $version                  ====\n"
    conda create --name bystro python=$version -y -q
    conda activate bystro
fi

echo -e "\n====                         Installing Python requirements                             ====\n"
find . -name 'requirements.txt' -exec pip install -q -r {} \;
echo -e "\n====                                       Done                                         ====\n"
