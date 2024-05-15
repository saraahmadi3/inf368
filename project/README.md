To use our trained agent make sure you have the file with our trained weights **cookie_patrol_agent.pth** and our agent **CookieAgent.py**.

in addition you will need to have all of these requirements installed:
numpy, torch, gymnasium

to use the trained aagent run the following code with your paths:

```
from CookieAgent import CookieAgent

agent = CookieAgent(save_path='cookie_patrol_agent.pth')
```

or 

```
from CookieAgent import CookieAgent

agent = CookieAgent()
agent.load('cookie_patrol_agent.pth')
```

to use it in an environment run the following code with your environment:

```
from CookieAgent import CookieAgent
import gymnasium as gym

agent = CookieAgent(save_path='cookie_patrol_agent.pth')
env=gym.make('your-environment')
state = env.reset()[0]


action = agent.select_action(state)
next_state, reward, done, terminated, info = env.step(action)
```

**NB** the way we initialise the max width and lifetime, is through the first pass of the select_action() method. this means that for each new environent that the agent is supposed to be tested on, it is requred to initiate a new CookieAgent with the weights from cookie_patrol_agent.pth