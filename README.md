# berryboxai

## Setup

1. Install Conda (if not already installed):

    a. If possible, install conda from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.anaconda.com/miniconda/)
    
    b. If using a USDA-ARS machine, use the following instructions:
    1. Open the "Software Center" app (Windows) or "Self Service" app (MacOS)
    2. Search for "Anaconda" and click "Install." Note that each user must install Anaconda separately.
    3. Wait for installation to complete (it may take a while).

2. Clone the repository:
    + Option 1: Use Git
        + Open up Powershell (Windows) or Terminal (Mac) and type the following:
            ```
            git clone https://github.com/neyhartj/berryboxai.git
            ```
    + Option 2: Download the repository as a .zip file
        + [Download the .zip file](https://github.com/neyhartj/berryboxai/archive/refs/heads/main.zip)
        + Move the .zip file to a suitable location, like the Documents folder.
        + Unzip the .zip file

3. Open up Powershell (Windows) or Terminal (Mac) and navigate to the unzipped folder:
    ```
    cd /path/to/unzipped/berryboxai/
    ```
    > Remember to replace "/path/to/unzipped" with the true path to the folder. 

4. Create a Conda environment by typing the following:
    ```
    conda env create -f environment.yml
    ```

5. Activate the environment:
    ```
    conda activate berryboxai_env
    ```

6. Install the package
    ```
    python setup.py install
    ```
