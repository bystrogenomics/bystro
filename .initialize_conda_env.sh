# Follow https://docs.conda.io/en/latest/miniconda.html to install miniconda
# Call this using "source .initialize_conda_env.sh"

if { conda env list | grep 'bystro'; } >/dev/null 2>&1; then
    echo -e "\n====Environment exists (bystro)====\n"
else
    conda create --name bystro python=3.11.3
    conda activate bystro
    pip install -r requirements.txt
fi

echo -e "\n====Activating environment (bystro) and installing requirements====\n"
conda activate bystro
conda install python=3.11.3 -y -q
find . -name 'requirements.txt' -exec pip install -r {} \;
