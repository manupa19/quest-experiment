## Installation
Make a virtual or conda environment and install the requirements.
```bash
pip install -r requirements.txt
```


## Run experiments
Activate the environment with requirements installed and run the following command.
```bash
python run.py
```
In run.py you can specify which algorithms to choose for the experiments.
Three standard procedures and Quest are given as a starting point.
Different data sets and parameters of Quest can be changed in config.yaml in the config folder.


## UI for tracking experiments
Your experiments are tracked in mlflow. Use the following command to open the mlflow dashboard.
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 7070

```

# IMPORTANT INFO! 
### Bug report #1
There is a little bug! The code gets stuck whenever QUEST is being executed. 
In order for it to finish hit Cltr+C or press the stop button in your IDE.
Although counter-intuitive, this somehow triggers the code. 
Hitting Ctrl+C or the stop button two times cancels the code.
### Bug report #2
A minor bug, there are always two log files produced, one legit, the other empty. 
Reason for this bug is known but solution was not implemented yet.