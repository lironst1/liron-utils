### Install New Environment
To install a new Anaconda environment, open Terminal and run:\
`bash "Others/New Python Environment/new_env.sh"`

* To create a new environment without the default packages set by the `.condarc` file, run:\
`conda create --name <env_name> --no-default-packages`

* To remove the environment, run:\
`conda remove --name <env_name> --all`

Then, restart PyCharm and change the PyCharm Python interpreter in Settings > Project > 
Python Interpreter > Add Interpreter > Conda Environment > Use Existing Environment > "MYENV" > Apply.

### pip
To install using pip, run:\
`pip install -r "/Users/lironst/Code Projects/Home/Others/pip requirements.txt" -U --progress-bar on`


To save the `requirements` file, run:\
`pip freeze => "/Users/lironst/Code Projects/Home/Others/pip requirements.txt"`
