# Install New Environment

## Install pyenv and Python version

```bash
brew install pyenv
brew install pyenv-virtualenv
pyenv init
```

Add the following to your shell configuration file (`~/.zshrc`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
pyenv activate "myvenv"
```

Then, install the desired Python version and set it as the global default:

```bash
pyenv virtualenv 3.11.14 myvenv  # Change to desired version
pyenv global 3.11.14/envs/myvenv
```

Then, choose one of the following package management tools to create a new environment and install dependencies.

## Poetry

1. Create a new virtual environment using pyenv-virtualenv:\
   `pyenv virtualenv 3.11.14 "myenv"`\
   Activate the environment: `pyenv activate "myenv"`

1. To install Poetry, run:

```bash
curl -sSL https://install.python-poetry.org | python -  # (Linux/Mac) 
```

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -  # (Windows PowerShell)
```

2. Add Poetry to the PATH:
    ```bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    ```
   In Windows, it installs to: `%APPDATA%\Python\Scripts`.

3. Add the `export` plugin:\
   `poetry self add poetry-plugin-export`


4. To create new `pyproject.toml` and `poetry.lock` files, run:
    ```bash
   poetry init
    ```
   Specify dependencies and versions in the `[projecet]` section, or use `poetry add <package>`.

5. To update dependencies, run:
    ```bash
    poetry update
    ```

6. To export the dependencies to a `requirements.txt` file, run:
    ```bash
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```
   You can then install the dependencies using pip.

7. To clear the cache, run:
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
