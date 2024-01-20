### Install New Environment
To install a new Anaconda environment, open Terminal and run:\
`bash "Others/New Python Environment/new_env.sh"`


Then, restart PyCharm and change the PyCharm Python interpreter in Settings > Project > 
Python Interpreter > Add Interpreter > Conda Environment > Use Existing Environment > 
"MYENV" > Apply.

### pip
To install using pip, run:\
`pip install -r "/Users/lironst/Code Projects/Home/Others/pip requirements.txt" -U --progress-bar on`


To save the requirements file, run:\
`pip freeze => "/Users/lironst/Code Projects/Home/Others/pip requirements.txt"`
