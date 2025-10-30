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
