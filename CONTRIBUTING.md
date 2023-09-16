# Contributing to Bystro

We welcome 3rd-party contributions. Out of respect for your time, please open an issue and confirm
the basic approach / design with the core team before submitting a PR. Please also see our code
[style guide](docs/style_guide.md).

We use a forking model of contribution, where you fork the repository, checkout a branch in your fork, and use GitHub's UI to issue the pull request back to main repository.

- https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model
- https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork

As an example, let's say we have forked the repository, and that fork has address `git@github.com:foobar/bystro.git`

The workflow is as usual, except you use your fork as the origin

```sh
# Create a local copy of the foobar fork of bystro.
# You will now have an "origin" remote repository record (check via git remote --verbose).
git clone git@github.com:foobar/bystro.git

# Create a new local branch named feature/some_feature.
git checkout -b feature/some_feature

# Assuming you made some changes to the code in the branch, commit those changes.
# Note that if you added files you will need to first `git add <path to file>`.
git commit -a -m "Addresses issue #NNN".

# Push to the remote repository, explicitly telling git that the remote repository you want to push to is "origin" (the foobar fork).
git push --set-upstream origin feature/some_feature
```

Then, visit your fork, and use GitHub to issue the pull request. GitHub will automatically set the bystrogenomics/master as the branch you're PR'ing against.

- https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork

To keep your fork up to date with main bystro repository, in your local clone create an `upstream` remote, that points to the main repository:

```sh
# Create an "upstream" remote repository record that points to the main (bystrogenomics) repository.
# The name you choose for the remote has no special significance, choose the name that makes the most sense to you!
# To check your remote definitions at any time, execute `git remote --verbose` in your terminal, in the bystro folder.
git remote add upstream git@github.com:bystrogenomics/bystro.git

# Fetch all of the branches, updates from the main repository bystrogenomics.
git fetch upstream
```

Often you may be interested in ensuring your forked repository has an up to date `master` branch:

```sh
# Ensure we're on our local master branch copy.
git checkout master

# Fetch the lastest changes from the master branch on the bystrogenomics main repo, and merge those changes into your local copy.
git pull upstream master
```

# Reviewing pull requests

Let's say a contributor has created a pull request, and we wish to review it. That pull request has number #999.

In our local clone:

```sh
# Create a new local branch, pr_999, that points to the pull request #999.
git fetch upstream pull/999/head:pr_999

# Enter our new branch.
git checkout pr_999
```

To fetch updates to the pull request (say the contributor updated their PR)

```sh
# Ensure we are on our local pr_999 branch.
git checkout pr_999

# Bring our local branch up to date with the latest changes in the pull request.
git pull upstream refs/pull/999/head
```

# Pull Request Etiquette

1. Keep pull requests small: < 500 maximum added lines of code with rare exceptions, < 200 lines preferred.
2. 1 commit per pull request: this will make commit history in the main branch easier to follow.

- Making significant contributions in a single commit is hard.
- To solve this, we can leverage the fact that we are operating in a fork of the main repository, and so rewriting our feature branch history has no chance of polluting the main repository's history.
- Let's say we are in our local clone of our bystro repository fork, on our feature branch feature/999. This feature branch has 7 commits, and we wish to condense that into 1:

  ```sh
  # Ensure we're on our local feature/999 branch.
  git checkout feature/999

  # Ensure we have the latest data from our fork.
  git pull origin feature/999

  # Keep all code changes, but roll back commit history 7 commits.
  git reset --soft HEAD~7

  # Create a new commit, with the cumulative changes of the past 7 commits.
  # This will overwrite your local feature/999 branch's history, such that the last 7 commit records are replaced with a single commit that has the cumulative changes of those last 7 commits.
  git commit -a -m "Fixes performance regression, addressing issue #100"

  # Double check that remote origin points to "foobar/bystro.git"
  git remote --verbose

  # Overwrite your fork's remote history to mirror the local history (with the past 7 commits condensed into 1).
  # With --force-with-lease, the push will be rejected if there are new commits on the remote branch that you have not pulled.
  git push --force-with-lease origin feature/999
  ```

- This will ensure that git history remains clean, while still allowing you to commit frequently during development.
