# Install New Environment

## Install pyenv and Python version

```bash
brew install pyenv
brew install pyenv-virtualenv
pyenv init
```

Add the following to the shell configuration file (`~/.zprofile`, `~/.zshrc`, `~/.bashrc`, etc.):

```bash
sudo nano ~/.zprofile
sudo nano ~/.zshrc

# Added by pyenv
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
```

Then, install the desired Python version and set it as the global default:

```bash
pyenv install 3.11.14  # Change to desired version
pyenv global 3.11.14
```

* Note: you can also create a new virtual environment with a specific Python version using:
    ```bash
    pyenv virtualenv 3.11.14 myvenv
    pyenv global 3.11.14/envs/myvenv
    ```

In Windows, just install [Python](https://www.python.org/downloads/windows/), then create a virtual environment:

```powershell
python -m venv C:\Users\liron\.virtualenvs\myvenv
```

Then, choose one of the following package management tools to create a new environment and install dependencies.

## Poetry

1. To install Poetry, run:
    ```bash
    curl -sSL https://install.python-poetry.org | python -  # (Linux/Mac) 
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -  # (Windows PowerShell)
    ```

2. Add Poetry path to the shell configuration file:
    ```bash
   # Added by Poetry
    export PATH="$HOME/.local/bin:$PATH"
    ```
   In Windows, it installs to: `%APPDATA%\Python\Scripts`.

3. Add plugins:
    ```bash
    poetry self add poetry-plugin-export
    poetry self add poetry-plugin-shell
    ```

4. Create new environment:
    ```bash
    cd "./liron_utils/New Python Environment"
   poetry env use python
   poetry update
    ```

5. To export the dependencies to a `requirements.txt` file, run:
    ```bash
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```
   You can then install the dependencies using pip.

6. To clear the cache, run:
    ```bash
    poetry cache list
    poetry cache clear --all [pypi, _default_cache, ...]
    ```

## Anaconda

To install a new Anaconda environment, open Terminal and run:
`bash "new_env.sh"`.
If using windows, open git bash and run the above line.

* To create a new environment without the default packages set by the `.condarc` file, run:\
  `conda create --name <env_name> --no-default-packages`

* To remove the environment, run:
  `conda remove --name <env_name> --all`

* To roll back an environment to its initial state, run:
  `conda install --rev 0`

Then, restart PyCharm and change the PyCharm Python interpreter in Settings > Project >
Python Interpreter > Add Interpreter > Conda Environment > Use Existing Environment > "MYENV" > Apply.

To clear the cache, run:\
`conda clean --all -y`

## pip

Get version requirements from `pip_requirements_in.txt` and save them to `pip_requirements.txt` using pip-tools:\
`pip-compile pip_requirements_in.txt --max-rounds 100 --output-file pip_requirements.txt`

Create a new virtual environment:\
`python -m venv "myvenv"`

Activate the virtual environment:\
`source "myvenv/bin/activate"` (Linux/Mac)\
`"myvenv\Scripts\activate.bat"` (Windows CMD)\
`& "myvenv\Scripts\Activate.ps1"` (Windows PowerShell)

To install using pip, run:\
`python -m pip install -r "requirements.txt" -U --progress-bar on`

To save the `requirements` file, run:\
`pip freeze => "pip_requirements.txt"`

To clear the cache, run:\
`pip cache purge`
