## How to contribute to this project

This will showcase a list of best practices, what to avoid and how to write your issues and PR to be easier to understand and merge.

### Git rules
  - pull frequently from main to avoid conflicts
  - rebase to solve conflicts locally
  - never push directly to main

### Conventions
Every commit is different and serves a specific purpose, I strongly recommend to write a placeholder firstly. Some examples of placeholders can be found [here](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/):

In short your commit should start with `[action]: message`. 

A good example is: `feat: implemented feature x` or `refactor: extracted functions into their proper place`

At the end of your work, if you have done multiple of these actions (fixing, refactor and feature), you may want to separate them better before commiting.

### Code quality and project structure
  - follow the python conventions
  - be consistent across the codebase
  - use clear variable names (not `x`, `car123`)
  - avoid printing out in the console, one should use the logging library for that
  - avoid comments that does not describe what is actually doing ( if you call a function that is called `check_for_unimplemented_features` don't write a comment above saying "checking for unimplemented feature", the code is already clear.
  - every file should have a clear purpose, don't mix and match functions across multiples python modules
  - do not commit any sensitive information, try to use `.gitignore`
  - please respect the structure of the project as much as possible

### Before submitting a PR
  - make sure the branch you worked on does not have conflicts with the target branch ( usually main )
  - the code compiles and runs as expected
  - you have done minimum testing on the funtionality implemented
  - don't leave debug code or random files
  - the most important step, **link the OpenProject task in the PR**

### Pull Request template

Here one can find a template with things that should appear in a PR.

Basically your pull request should contain the following information:

The first line and the most important is the reference to the OpenProject task number. Please write as the first line the following structure:

```
OP#xxx
```

IMPORTANT!
Match the number of your task in the OpenProject with the one in the description. Examples: (if your task is 435, write `OP#435`, don't write (`xxx` or `123`), these are just placeholder that serve only examples)

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

A template is provided by default when creating a new Pull Request, you may either just complete it as is or remake it to fit your preferences as long as it respects the requirements.