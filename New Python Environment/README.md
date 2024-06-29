### Install New Environment
To install a new Anaconda environment, open Terminal and run:\
`bash "new_env.sh"`\
If using windows, open git bash and run the above line.

* To create a new environment without the default packages set by the `.condarc` file, run:\
`conda create --name <env_name> --no-default-packages`

* To remove the environment, run:\
`conda remove --name <env_name> --all`

Then, restart PyCharm and change the PyCharm Python interpreter in Settings > Project > 
Python Interpreter > Add Interpreter > Conda Environment > Use Existing Environment > "MYENV" > Apply.

### pip
To install using pip, run:\
`pip install -r "pip_requirements.txt" -U --progress-bar on`


To save the `requirements` file, run:\
`pip freeze => "pip_requirements.txt"`
