# Contributing to Bystro
We welcome 3rd-party contributions.  Out of respect for your time, please open an issue and confirm
the basic approach / design with the core team before submitting a PR.  Please also see our code
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
git push --set-upstream origin feature/999
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


If you have multiple commits in your feature branch already (the branch you are PR'ing), you can squash those changes:
  ```sh
  # Ensure we're on our local feature/999 branch.
  git checkout feature/999
  
  # Bring our local branch up to date with our remote branch
  # If the branch's remote is "origin" this will be equivalent to executing: git pull origin feature/999
  git pull
  ```
  
  To squash all commits in the feature branch:
  ```sh
  # Squashes all commits in the feature branch
  # Here we are assuming the bystrogenomics repository is the "upstream" remote
  # e.g., that you ran `git remote add upstream git@github.com:bystrogenomics/bystro.git`
  git reset --soft upstream/master
  ```

  Then, give your squashed commits a message, and push to remote:
  ```sh
  # Give your squashed commit a message
  git commit -a -m "Fixes performance regression, addressing issue #100"
  
  # Overwrite your fork's remote history to mirror the local history (with the past 7 commits condensed into 1).
  # With --force-with-lease, the push will be rejected if there are new commits on the remote branch that you have not pulled.
  git push --force-with-lease
  ```

  If you have already created a branch from feature/999, which was just squashed, you can bring a child branch (feature/1000) up to date by:
  ```sh
  # feature/1000 was branched from feature/999 before the squash
  git checkout feature/1000

  # bring feature/1000 up to date with feature/999
  git rebase --fork-point feature/999

  # push the changes to feature/1000 to remote
  git push --force-with-lease
  ```

  If you now want to squash the commits that are exclusive to feature/1000:
  ```sh
  # Checkout the child branch (feature/1000), and bring it up to date with the changes in the parent branch (feature/999)
  git checkout feature/1000
  git merge feature/999

  # This command will go place all of the code changes that are exclusive to feature/1000
  # into an uncommitted state
  git reset --soft feature/999

  # Commit all of the changes, thereby squashing them
  git commit -a -m "some commit message for the commits exclusive to feature/1000 that have been squashed"

  # Push to the feature branch
  git push --force-with-lease
  ```

This will ensure that git history remains clean, while still allowing you to commit frequently during development.
