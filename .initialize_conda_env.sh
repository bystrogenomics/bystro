# Follow https://docs.conda.io/en/latest/miniconda.html to install miniconda
# Call this using "source .initialize_conda_env.sh"
version="3.11.3"
if { conda env list | grep 'bystro'; } >/dev/null 2>&1; then
    echo -e "\n====Bystro environment exists, activating and ensuring up to date with Python $version====\n"
    conda activate bystro
    conda install python=$version -y -q
else
    echo -e "\n====Creating Bystro environemnt with Python $version====\n"
    conda create --name bystro python=$version -y -q
    conda activate bystro
    pip install -r requirements.txt
fi

echo -e "\n====Installing Python requirements====\n"
find . -name 'requirements.txt' -exec pip install -r {} \;
