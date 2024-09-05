# TowerScout Databricks library for Python/PySpark

TowerScout detects cooling towers from aerial imagery. This library contains functions, classes, and abstractions for the [Databricks][databricks-docs]/[PySpark][pyspark-docs] implementations needed for the project's backend. 

## Getting started

### Install the package
- We store all versions of this library in the TowerScout Databricks volumes, located under `edav-dev-csels`

## Next steps

### Additional documentation


## Contributing

To maintain consistency and ensure high quality code, please follow these guidelines when contributing.

**1. Branches**
* We use the GitHub Flow branching style. No code should be committed to main. Branches should be prepended with `feature`, `bugfix`, `hotfix`, `release`, `test`, `experiment`, `docs`, `refactor`, or `ci`. Additional descriptors may be added over time. Your branch name must include the JIRA ticket associated with it in the format `<prefix>/JIRA-123-what-am-i-doing`. For example, `experiment/DMAC-300-efficientnet-latent-dim`

**2. Code Style**

* All python code must adhere to the [PEP 8][pep8-docs] style guide. We also enforce this using [Pylint][pylint-docs] to ensure code quality. Please run Pylint on your code before submitting any changes.

**3. Type Hints**

* We use [Mypy][mypy-docs] for static checking. Pleasure ensure that all functions and methods include proper type hints to pass Mypy checks. 

**4. Modular Design**

* Functions and logic should be written as isolated modules, separate from any PySpark notebooks. 
* Notebooks should only import and use these functions, ensuring the notebooks remain clean and focused on orchestrating tasks. 
* When submitting changes, please move any functions or logic to an appropriate place in the `src` directory. 

**5. Unit Tests**

* Comprehensive unit tests are required for all new features and bug fixes. Please write your tests using [Pytest][pytest-docs], ensuring they cover edge cases and are isolated from external dependencies. 
* Tests are run on Databricks with access to Spark. 

**6. Submitting Changes:**

* Ensure your code passes all linting and type-checking tools. Databricks has modules for these.

* Write or update tests to cover any new functionality or to validate bug fixes.

* Open a pull request with a clear description of the changes you've made and how they contribute to the project. 

### Directories
* **development**: A directory for notebooks used to develop. This directory is not maintained for long periods of time and is only provided as a convenience. You **SHOULD NOT** submit code in this directory for your PR's. These notebooks can be **converted** into **actual** examples if you want to keep a history

* **examples**: A directory for example usage of functions or classes you've created. These notebooks should include markdown and comments to outline what you're doing and why you're doing it. Examples should be written for someone with Python development experience, but possibly not PySpark.

* **src**: This directory is packaged and installed onto TowerScout Databricks clusters. If you want your code available in these environments, it goes here as code files. 

* **tests**: Unit tests for code in the `src` directory

* **workflows**: Directory for notebooks and YAML/JSON workflow files. Notebooks in this directory **should only** contain orchestration logic.

## Gotchas

### MLFlow RESOURCE_DOES_NOT_EXIST
When working with Unity Catalog MLFlow, you may encounter an error message "RESOURCE_DOES_NOT_EXIST". You need to ensure your CDC account and the TowerScout service principal has access to the directory you're running the code on. You can do this by clicking "Share" and adding the appropriate accounts. 


<!-- LINKS -->
[databricks-docs]: https://learn.microsoft.com/en-us/azure/databricks/
[mypy-docs]: https://mypy.readthedocs.io/en/stable/getting_started.html
[pep8-docs]: https://peps.python.org/pep-0008/
[pip-docs]: https://pypi.org/project/pip/
[pylint-docs]: https://docs.pylint.org/
[pytest-docs]: https://docs.pytest.org/en/stable/ 
[pyspark-docs]: https://spark.apache.org/docs/latest/api/python/index.html
[python-docs]: https://www.python.org/downloads/