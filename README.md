## Generating Company Summaries via Generative AI



## Prerequisites

The summaries are generated by a Jupyter notebook and requires that the following be installed first:

1. Install pyenv, this lets you install multiple Python interpreters. Follow [these instructions](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

2. Using pyenv, install Python 3.10.9. Other Python versions may work but have not been tested.

3. Install [poetry](https://python-poetry.org/docs/).

4. After cloning this repo, in the top-level of the repo create a poetry environment: `poetry install`. Then, to use this environment's Python interpreter: `poetry shell`

5. Create a kernel with these Python dependencies for Jupyter: `python3 -m ipykernel install --user --name generate-company-summary`

6. Create an account at [https://serpapi.com/](https://serpapi.com/) and add your SERP API key in the cell that contains `%env SERPAPI_API_KEY=...`

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
