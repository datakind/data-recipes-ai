# Contributing to DOT

Hi! Thanks for your interest in contributing to Data Recipes AI, we're really excited to see you! In this document we'll try to summarize everything that you need to know to do a good job.

## New contributor guide

To get an overview of the project, please read the [README](README.md) and our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable.


## Getting started
### Creating Issues

If you spot a problem, [search if an issue already exists](https://github.com/datakind/data-recipes-ai/issues). If a related issue doesn't exist, 
you can open one there, selecting an appropriate issue type.

As a general rule, we don’t assign issues to anyone. If you find an issue to work on, you are welcome to open a PR with a fix.

## Making Code changes

## Setting up a Development Environment

To set up your local development environment for contributing follow the steps
in the paragraphs below.

The easiest way to develop is to run in the Docker environment, see [README](./README.md) for more details. 

## Code quality tests

The repo has been set up with black and flake8 pre-commit hooks. These are configured in the ``.pre-commit-config.yaml` file and initialized with `pre-commit autoupdate`.

On a new repo, you must run `pre-commit install` to add pre-commit hooks.

To run code quality tests, you can run `pre-commit run --all-files`

## Tests

You should write tests for every feature you add or bug you solve in the code.
Having automated tests for every line of our code lets us make big changes
without worries: there will always be tests to verify if the changes introduced
bugs or lack of features. If we don't have tests we will be blind and every
change will come with some fear of possibly breaking something.

For a better design of your code, we recommend using a technique called
[test-driven development](https://en.wikipedia.org/wiki/Test-driven_development),
where you write your tests **before** writing the actual code that implements
the desired feature.

You can use `pytest` to run your tests, no matter which type of test it is.

## GitHub Workflow

As many other open source projects, we use the famous
[gitflow](https://nvie.com/posts/a-successful-git-branching-model/) to manage our
branches.

Summary of our git branching model:

- Get all the latest work from the upstream repository (`git checkout main`)
- Create a new branch off with a descriptive name (for example:
  `feature/new-test-macro`, `bugfix/bug-when-uploading-results`). You can
  do it with (`git checkout -b <branch name>`)
- Make your changes and commit them locally  (`git add <changed files>>`,
  `git commit -m "Add some change" <changed files>`). Whenever you commit, the self-tests 
  and code quality will kick in; fix anything that gets broken
- Push to your branch on GitHub (with the name as your local branch:
  `git push origin <branch name>`). This will output a URL for creating a Pull Request (PR)
- Create a pull request by opening the URL a browser. You can also create PRs in the GitHub
  interface, choosing your branch to merge into main
- Wait for comments and respond as-needed
- Once PR review is complete, your code will be merged. Thanks!!

### Tips

- Write [helpful commit
  messages](https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message)
- Anything in your branch must have no failing tests. You can check by looking at your PR
  online in GitHub
- Never use `git add .`: it can add unwanted files;
- Avoid using `git commit -a` unless you know what you're doing;
- Check every change with `git diff` before adding them to the index (stage
  area) and with `git diff --cached` before committing;
- If you have push access to the main repository, please do not commit directly
  to `dev`: your access should be used only to accept pull requests; if you
  want to make a new feature, you should use the same process as other
  developers so your code will be reviewed.

## Code Guidelines

- Use [PEP8](https://www.python.org/dev/peps/pep-0008/);
- Write tests for your new features (please see "Tests" topic below);
- Always remember that [commented code is dead
  code](https://www.codinghorror.com/blog/2008/07/coding-without-comments.html);
- Name identifiers (variables, classes, functions, module names) with readable
  names (`x` is always wrong);
- When manipulating strings, we prefer either [f-string
  formatting](https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals)
  (f`'{a} = {b}'`) or [new-style
  formatting](https://docs.python.org/library/string.html#format-string-syntax)
  (`'{} = {}'.format(a, b)`), instead of the old-style formatting (`'%s = %s' % (a, b)`);
- You will know if any test breaks when you commit, and the tests will be run
  again in the continuous integration pipeline (see below);
