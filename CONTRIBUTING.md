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


# Code review
Code review has many purposes.  Most of all, code review helps to
ensure that the submitted code is:

- necessary (the problem is worth solving and cannot be solved by any easier method)
- appropriately designed (the high-level design is a reasonable approach to solving the problem)
- correct (the code faithfully implements the high leven design)
- clear (the code is written so that it's easy to read, modify and extend)

Review also helps to disseminate knowledge of the codebase, and of
engineering practices in general.  Lastly, review serves as a kind of
simulated dry run for code maintenance.  Difficulties that the
reviewer encounters while reading the code are likely to be
difficulties that the next engineer will have when modifying it.

https://storage.googleapis.com/pub-tools-public-publication-data/pdf/80735342aebcbfc8af4878373f842c25323cb985.pdf

## Reviewer's guide to code review
https://google.github.io/eng-practices/review/reviewer/

### The Standard of Code Review

`In general, reviewers should favor approving a CL once it is in a
state where it definitely improves the overall code health of the
system being worked on, even if the CL isn’t perfect.`

#### Mentoring
#### Principles
#### Resolving Conflicts
### What to Look For In a Code Review
#### Design
#### Functionality
#### Complexity
#### Tests
#### Naming
#### Comments
#### Style
#### Consistency
#### Documentation
#### Every Line
#### Context
#### Good Things
### Navigating a CL in Review
#### Take a Broad View of the Change
#### Examine the Main Parts of the Change
#### Look Through the Rest of the CL in an Appropriate Sequence
### Speed of Code Reviews
#### Why Should Code Reviews be fast?
#### How Fast Should Code Reviews be?
#### Speed vs. Interruption
#### Fast Responses

#### LGTM With Comments
Sometimes an approval will be given with certain changes requested.  The changes fall into two categories:

1. Changes that are required before the PR should be merged
2. Changes that are merely recommended, but not required

The reason for the first type of requested change is to avoid an additional round of code review: the
reviewer *conditionally approves* the PR, subject to the requested changes being made: once the author
makes those changes, they're free to merge.  The second set of changes are merely recommendations which
the author can implement or not before merging.

It is incumbent upon the reviewer to clearly indicate whether their approval of the PR is conditional on
any changes requested.

It is incumbent upon the author not to abuse a conditional approval by merging the PR without making the
requested changes.  If the author disagrees with the requested changes, this disagreement should be
resolved before merging.


#### Large CLs

#### Code Review Improvements Over Time

> If you follow these guidelines and you are strict with your code reviews, you should find that the
> entire code review process tends to go faster and faster over time. Developers learn what is
> required for healthy code, and send you CLs that are great from the start, requiring less and less
> review time. Reviewers learn to respond quickly and not add unnecessary latency into the review
> process. But don’t compromise on the code review standards or quality for an imagined improvement in
> velocity—it’s not actually going to make anything happen more quickly, in the long run.



### How to Write Code Review Comments
How to write code review comments
#### Summary
- Be kind.
- Explain your reasoning.
- Balance giving explicit directions with just pointing out problems and letting the developer decide.
- Encourage developers to simplify code or add code comments instead of just explaining the complexity to you.
#### Courtesy
- Be courteous and professional
- Stick to commenting about the code, rather than about the person who wrote it.
- Always remember that you might just be misunderstanding.  But also strive to find ways to reduce future misunderstandings.
#### Explain Why
Try to give context and explanation for why you're making your comments.

#### Giving Guidance
- In general, it's the author's ultimate responsibility to fix a PR, not the reviewer's.
- But we should always look for opportunities to get a high return on investment by spending a little of our own time in order to save a lot of our colleagues.
- Aim to strike a balance between giving direct instructions vs. pointing out a problem and advising the author on the options to fix it.
- Praise what the author did particularly well in addition to identifying problems in code.

#### Label comment severity
Make it clear whether you consider your requested change mandatory for approval.  Use the preface
`nit:` or similar language in order to make clear that a requested change is optional.

#### Accepting Explanations
If you ask for an explanation of a piece of code, *strongly* prefer that the explanation be somehow
baked into the PR, and not simply as a reply in the code review tool.

The best response to a request for clarification is usually a rewriting of that code to make the
reviewer's question unnecessary.  Rewriting code includes refactoring, but also renaming of variable
or function names, expansion of docstrings, additional tests, updated documentation, &c.

Sometimes the author's reply to a reviewer question can simply be copied and pasted into the code as
a code comment.  This is most appropriate when the comment addresses *why* a piece of code does what
it does, and not merely *how* it does it.

Rarely, a direct reply to a reviewer in the code review tool may be appropriate.  This is usually
only true when the author is communicating information that is unknown to the reviewer, but could be
normally expected of anyone reading the code.  Bear in mind, though, that in a small organization
the reviewer is typically *ipso facto* representative of someone who will read the code in the
future.

### Handling Pushback in Code Reviews
#### Who is right?
- The author is typically closer to the code than the reviewer and may have insight into the problem that the reviewer lacks
- But the reviewer, by virtue of being further away from the code, is often a better judge of how the code will appear to engineers who will have to maintain it later.
- Don't be afraid to continue to advocate for changes if the improvement to code health justifies the work required.

#### Upsetting Developers
- Reviewers naturally worry about upsetting authors by requesting changes
- Sometimes this happens, but it's typically short-lived and authors are usually thankful later that the code is improved
- Remain polite and friendly and keep requests for changes professional

#### On "Cleaning It Up Later"
> A common source of push back is that developers (understandably) want to get things done. They
> don’t want to go through another round of review just to get this CL in. So they say they will clean
> something up in a later CL, and thus you should LGTM this CL now.

- "Cleaning it up later" could, theoretically, work
- But in practice it's rare, because people are human and have other things to do, and scheduled maintenance of technical debt is never as exciting or rewarding as working on new features.
- It's also harder to come back to something and clean it up later than to clean it up now, because of context switches.
- A strategy of "cleaning it up later" is, empirically, "a common way for codebases to degenerate".
  Degenerating code health leads to low velocity and developer morale, with attendant feelings of
  frustation, avoidance and procrastination when working with that code.

#### General Complaints About Strictness

> If you previously had fairly lax code reviews and you switch to having strict reviews, some
> developers will complain very loudly. Improving the speed of your code reviews usually causes these
> complaints to fade away.

> Sometimes it can take months for these complaints to fade away, but eventually developers tend to
> see the value of strict code reviews as they see what great code they help generate. Sometimes the
> loudest protesters even become your strongest supporters once something happens that causes them to
> really see the value you’re adding by being strict.

#### Resolving Conflicts
Above all, refer to The Standard of Code Review: "does this PR definitely improve code health overall?" when resolving conflicts.

## Author's guide to code review
https://google.github.io/eng-practices/review/developer/
### Writing Good PR Descriptions

PR descriptions should take the following format:

```
Summarize the PR in the first line in one complete sentence using the imperative mood.

Elaborate in one or more additional paragraphs below.  Describe the background context for the change,
i.e. what motivates the PR in the first place.  Describe the high-level approach taken.  Note any other 
relevant details or concerns, such as additional manual testing that was performed.  If the PR is
part of a series of planned, PRs, note that as well.  

Closes #NNN (Each PR should address an issue, and tagging the issue number in the PR will automatically
close that issue.)

```
### Small PRs
Small, simple CLs are:


> - Reviewed more quickly. It’s easier for a reviewer to find five minutes several times to review small CLs than to set aside a 30 minute block to review one large CL.
> - Reviewed more thoroughly. With large changes, reviewers and authors tend to get frustrated by large volumes of detailed commentary shifting back and forth—sometimes to the point where important points get missed or dropped.
> - Less likely to introduce bugs. Since you’re making fewer changes, it’s easier for you and your reviewer to reason effectively about the impact of the CL and see if a bug has been introduced.
> - Less wasted work if they are rejected. If you write a huge CL and then your reviewer says that the overall direction is wrong, you’ve wasted a lot of work.
> - Easier to merge. Working on a large CL takes a long time, so you will have lots of conflicts when you merge, and you will have to merge frequently.
> - Easier to design well. It’s a lot easier to polish the design and code health of a small change than it is to refine all the details of a large change.
> - Less blocking on reviews. Sending self-contained portions of your overall change allows you to continue coding while you wait for your current CL in review.
> - Simpler to roll back. A large CL will more likely touch files that get updated between the initial CL submission and a rollback CL, complicating the rollback (the intermediate CLs will probably need to be rolled back too).
> - Note that reviewers have discretion to reject your change outright for the sole reason of it being too large. Usually they will thank you for your contribution but request that you somehow make it into a series of smaller changes. It can be a lot of work to split up a change after you’ve already written it, or require lots of time arguing about why the reviewer should accept your large change. It’s easier to just write small CLs in the first place.

### What is Small?
- The ideal PR size is: "one self-contained change"
- Include relevant tests and documentation in the same PR.
- But don't make PRs so small that their implications are difficult to understand.
- Large PRs that result from automated tools (e.g. formatters) are OK, but should never mix with human-written changes within a PR.
### How to Handle Reviewer Comments
#### Don't Take it Personally
#### Respond through the code
#### Think Collaboratively
- Make sure you understand what the reviewer is asking for before responding, and ask for clarification if unsure.
- If you understand but disagree, respond in a way that drives the code review process towards resolution
  instead of shutting it down:
  ```
  Bad: “No, I’m not going to do that.”

  Good: “I went with X because of [these pros/cons] with [these tradeoffs]. My understanding is that using Y would be 
  worse because of [these reasons]. Are you suggesting that Y better serves the original tradeoffs, that we should 
  weigh the tradeoffs differently, or something else?”
```
  
- Be courteous and respectful, and try to bear in mind that both author and reviewer have the same
  interests in incorporating the author's contributions while maintaining the overall code health of the
  system.
#### Resolving Conflicts
- If a quick consensus can't be reached through the code review interface, schedule a meeting or call to
  discuss verbally.  Record the relevant decisions from that call in the code review tool so as not to
  give the appearance of a long, contentious discussion with no resolution.
- If consensus can't be achieved after a call, ask for a tie-breaking recommendation from another
  developer, preferably someone with experience relevant to the PR.  In no case should PRs lie in limbo
  because of unresolved disagreements between author and reviewer.
