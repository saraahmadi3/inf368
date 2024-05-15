# Code for final project

The environments for the problem assignment are provided in the file *bandits.py*. 

## Working with the bandit environment
The bandit environment is provided in the file *bandits.py*. 

Once you import the file, you can instantiate the environments **Bandits_final()**. You can then interact with this environment as you interacted with the MAB environment in the first problem set (see README.md for assignment 1).

## Working with the cookiedisaster environment
The cookiedisaster environment is provided in the file *cookiedisaster.py*. 

In order to setup the *cookiedisaster* environment,  you will have to perform the following steps:
1. Unzip *cookiedisaster.zip*
2. Move into the directory containing the *cookiedisaster/* folder
3. From the prompt, run the command  ```pip install -e cookiedisaster```

You will then be able to instantiate the environments by loading ```cookiedisaster-v1```, ```cookiedisaster-v2``` or ```cookiedisaster-v3``` as:
```
import gymnasium
import cookiedisaster

env = gymnasium.make('cookiedisaster-v1')
```

In the file ```BaseAgent.py``` you will find a simple interface for your RL agent. Please provide your trained agent following this interface (plus any configuration file that will be loaded by calling the ```load()``` method). Your agent will be tested on unseen *cookiedisaster* environments by plugging it into a loop and calling its ```select_action(obs)``` method.

## Working with PPO
The *InvertedPendulum* environment is part of the *gymnasium* library.

In order to setup load it, you can run:

```
import gymnasium

env = gymnasium.make('InvertedPendulum-v4')
```