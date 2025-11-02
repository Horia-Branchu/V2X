## Pull Request template

Here one can find a template with things that should appear in a PR.

Basically your pull request should contain the following information:

The first line and the most important is the reference to the OpenProject task number. Please write as the first line the following structure:

```
OP#xxx

```

IMPORTANT!
Match the number of your task in the OpenProject with the one in the description. Examples: (if your task is 435, write `OP#435`, don't write (`xxx` or `123`, these are just placeholder that serve only examples)

Leave a line empty after the reference and continue to the description.

The description should contain a brief explanation of what you did, what you encountered and a brief explanation of what you implemented throughout the task.
Example:

`
In this pull request, I managed to implement a base class of the Sumo Enviroment to run in different scenarios. I created an abstract class with abstract functions that need to be called in the RL module. Each scenario, TLS, BSM will have to inherit this class and implement their specific functions that fit best. `

Finally it should look like this:

---

OP#xxx

In this pull request, I managed to implement a base class of the Sumo Enviroment to run in different scenarios. I created an abstract class with abstract functions that need to be called in the RL module. Each scenario, TLS, BSM will have to inherit this class and implement their specific functions that fit best.

My thinking was the following: simulation_runner.py is a script that should be called in order to start the simulation. For the simulation to work we need the proper environment. Our environment will be dynamic in the sense of the features (TLS, BSM, etc). More, RL needs specific functions to be implemented (reward functions) so this raises the need for abstract classes so our features to be tailored for their scenarios. One may create different classes for TLS, BSM and so on and implement their quirks.

---
