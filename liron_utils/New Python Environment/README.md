# Install New Environment

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