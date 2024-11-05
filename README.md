# TowerScout

A tool for identifying cooling towers from satellite and aerial imagery

TowerScout Team:

<a target="_blank" href="https://www.linkedin.com/in/karenkwong/">Karen Wong</a>,
<a target="_blank" href="https://www.linkedin.com/in/jia-lu-gracie-a8b5a71a/">Jia Lu</a>,
<a target="_blank" href="https://www.linkedin.com/in/gunnarmein/">Gunnar Mein</a>,
<a target="_blank" href="https://www.linkedin.com/in/thaddeussegura/">Thaddeus Segura</a><br>

Licensed under <a target="_blank" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC-BY-NC-SA-4.0</a>
(see <a target="_blank" href="https://github.com/TowerScout/TowerScout/blob/main/LICENSE.TXT">LICENSE.TXT</a> in the root of the repository for details)


# Code Style and Formatting
## Pre-commit Hooks
Pre-commit hooks are useful for uniformly formatting code such as missing semicolons, removing trailing whitespace,
and standardizing line lengths. The goal is to allow the developer to focus on the code without getting bogged down in
trivial style nitpicks.

## Installation
`pip install pre-commit`

Run `pre-commit install` to setup the git hook scripts. Now `pre-commit` will run automatically on a `git commit`

Run `pip install pylint` to install
It's highly recommended to run pylint on any files you have edited before committing. PyLint helps standardize code
format and will warn of any issues. It will not throw an error.
