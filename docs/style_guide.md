# Style Guide

## Notes on Tooling
For Python development, Bystro uses `black` for formatting, `ruff` for linting, `mypy` for static
analysis, and `pytest` for unit-testing.  For best results, these tools should be run from the
python project top-level and called with their project-specific configuration files:

```
# make sure you're in a bystro-specific virtualenv
pip install -r $BYSTRO_ROOT/python/requirements-dev.txt
cd $BYSTRO_ROOT/python/python/bystro

ruff --config ~/python_linting_config/ruff.toml .
mypy --config-file $BYSTRO_ROOT/python/mypy.ini .
pytest .
```

### In Defense of Opinionated Tools

Opinionated tools, like the ones we use in this repository, offer significant advantages in error
prevention and code consistency. By automatically enforcing coding conventions, these tools save
time and mental effort for developers. They detect common static errors, promote best practices, and
ensure uniformity between developer workflows.  These tools work together to catch errors at
different levels, from simple syntax issues to more complex logic problems. They become even more
valuable as the codebase grows, providing essential checks in larger projects.


### The Right Balance

Finding the right balance between short-term and long-term velocity is crucial. We've curated our
linting ruleset starting with the default settings and customizing it to suit our specific concerns
in scientific programming, with the aim of allowing developers to write more efficient unit test
suites and focus on higher-level concerns like correctness and performance.

## Continual Improvement

We believe in continuous improvement. Our ruleset isn't set in stone, and we welcome feedback and
suggestions for refinement. If you feel that certain rules frustrate rather than enhance your
productivity, please let us know, and we'll be happy to reevaluate and make necessary adjustments.
We want our tooling to work for developers, not the other way around.

## Engage and Contribute

Your input and contributions are gratefully appreciated. If you encounter any issues or have ideas
to improve our workflow or ruleset, feel free to share your thoughts. We appreciate your dedication
to maintaining code quality and fostering a collaborative environment.

Thanks!
