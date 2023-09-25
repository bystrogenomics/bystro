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

```
If you follow these guidelines and you are strict with your code reviews, you should find that the entire code review process tends to go faster and faster over time. Developers learn what is required for healthy code, and send you CLs that are great from the start, requiring less and less review time. Reviewers learn to respond quickly and not add unnecessary latency into the review process. But don’t compromise on the code review standards or quality for an imagined improvement in velocity—it’s not actually going to make anything happen more quickly, in the long run.
```


### How to Write Code Review Comments
How to write code review comments
#### Summary
Be kind.
Explain your reasoning.
Balance giving explicit directions with just pointing out problems and letting the developer decide.
Encourage developers to simplify code or add code comments instead of just explaining the complexity to you.
#### Courtesy
In general, it is important to be courteous and respectful while also being very clear and helpful to the developer whose code you are reviewing. One way to do this is to be sure that you are always making comments about the code and never making comments about the developer. You don’t always have to follow this practice, but you should definitely use it when saying something that might otherwise be upsetting or contentious. For example:

Bad: “Why did you use threads here when there’s obviously no benefit to be gained from concurrency?”

Good: “The concurrency model here is adding complexity to the system without any actual performance benefit that I can see. Because there’s no performance benefit, it’s best for this code to be single-threaded instead of using multiple threads.”

#### Explain Why
One thing you’ll notice about the “good” example from above is that it helps the developer understand why you are making your comment. You don’t always need to include this information in your review comments, but sometimes it’s appropriate to give a bit more explanation around your intent, the best practice you’re following, or how your suggestion improves code health.

#### Giving Guidance
In general it is the developer’s responsibility to fix a CL, not the reviewer’s. You are not required to do detailed design of a solution or write code for the developer.

This doesn’t mean the reviewer should be unhelpful, though. In general you should strike an appropriate balance between pointing out problems and providing direct guidance. Pointing out problems and letting the developer make a decision often helps the developer learn, and makes it easier to do code reviews. It also can result in a better solution, because the developer is closer to the code than the reviewer is.

However, sometimes direct instructions, suggestions, or even code are more helpful. The primary goal of code review is to get the best CL possible. A secondary goal is improving the skills of developers so that they require less and less review over time.

Remember that people learn from reinforcement of what they are doing well and not just what they could do better. If you see things you like in the CL, comment on those too! Examples: developer cleaned up a messy algorithm, added exemplary test coverage, or you as the reviewer learned something from the CL. Just as with all comments, include why you liked something, further encouraging the developer to continue good practices.

#### Label comment severity
Consider labeling the severity of your comments, differentiating required changes from guidelines or suggestions.

Here are some examples:

Nit: This is a minor thing. Technically you should do it, but it won’t hugely impact things.

Optional (or Consider): I think this may be a good idea, but it’s not strictly required.

FYI: I don’t expect you to do this in this CL, but you may find this interesting to think about for the future.

This makes review intent explicit and helps authors prioritize the importance of various comments. It also helps avoid misunderstandings; for example, without comment labels, authors may interpret all comments as mandatory, even if some comments are merely intended to be informational or optional.

#### Accepting Explanations
If you ask a developer to explain a piece of code that you don’t understand, that should usually result in them rewriting the code more clearly. Occasionally, adding a comment in the code is also an appropriate response, as long as it’s not just explaining overly complex code.

Explanations written only in the code review tool are not helpful to future code readers. They are acceptable only in a few circumstances, such as when you are reviewing an area you are not very familiar with and the developer explains something that normal readers of the code would have already known.
### Handling Pushback in Code Reviews
Who is right?
When a developer disagrees with your suggestion, first take a moment to consider if they are correct. Often, they are closer to the code than you are, and so they might really have a better insight about certain aspects of it. Does their argument make sense? Does it make sense from a code health perspective? If so, let them know that they are right and let the issue drop.

However, developers are not always right. In this case the reviewer should further explain why they believe that their suggestion is correct. A good explanation demonstrates both an understanding of the developer’s reply, and additional information about why the change is being requested.

In particular, when the reviewer believes their suggestion will improve code health, they should continue to advocate for the change, if they believe the resulting code quality improvement justifies the additional work requested. Improving code health tends to happen in small steps.

Sometimes it takes a few rounds of explaining a suggestion before it really sinks in. Just make sure to always stay polite and let the developer know that you hear what they’re saying, you just don’t agree.

#### Upsetting Developers
Reviewers sometimes believe that the developer will be upset if the reviewer insists on an improvement. Sometimes developers do become upset, but it is usually brief and they become very thankful later that you helped them improve the quality of their code. Usually, if you are polite in your comments, developers actually don’t become upset at all, and the worry is just in the reviewer’s mind. Upsets are usually more about the way comments are written than about the reviewer’s insistence on code quality.

#### Cleaning It Up Later
A common source of push back is that developers (understandably) want to get things done. They don’t want to go through another round of review just to get this CL in. So they say they will clean something up in a later CL, and thus you should LGTM this CL now. Some developers are very good about this, and will immediately write a follow-up CL that fixes the issue. However, experience shows that as more time passes after a developer writes the original CL, the less likely this clean up is to happen. In fact, usually unless the developer does the clean up immediately after the present CL, it never happens. This isn’t because developers are irresponsible, but because they have a lot of work to do and the cleanup gets lost or forgotten in the press of other work. Thus, it is usually best to insist that the developer clean up their CL now, before the code is in the codebase and “done.” Letting people “clean things up later” is a common way for codebases to degenerate.

If a CL introduces new complexity, it must be cleaned up before submission unless it is an emergency. If the CL exposes surrounding problems and they can’t be addressed right now, the developer should file a bug for the cleanup and assign it to themselves so that it doesn’t get lost. They can optionally also write a TODO comment in the code that references the filed bug.

#### General Complaints About Strictness
If you previously had fairly lax code reviews and you switch to having strict reviews, some developers will complain very loudly. Improving the speed of your code reviews usually causes these complaints to fade away.

Sometimes it can take months for these complaints to fade away, but eventually developers tend to see the value of strict code reviews as they see what great code they help generate. Sometimes the loudest protesters even become your strongest supporters once something happens that causes them to really see the value you’re adding by being strict.

#### Resolving Conflicts
If you are following all of the above but you still encounter a conflict between yourself and a developer that can’t be resolved, see The Standard of Code Review for guidelines and principles that can help resolve the conflict.

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

```
- Reviewed more quickly. It’s easier for a reviewer to find five minutes several times to review small CLs than to set aside a 30 minute block to review one large CL.
- Reviewed more thoroughly. With large changes, reviewers and authors tend to get frustrated by large volumes of detailed commentary shifting back and forth—sometimes to the point where important points get missed or dropped.
- Less likely to introduce bugs. Since you’re making fewer changes, it’s easier for you and your reviewer to reason effectively about the impact of the CL and see if a bug has been introduced.
- Less wasted work if they are rejected. If you write a huge CL and then your reviewer says that the overall direction is wrong, you’ve wasted a lot of work.
- Easier to merge. Working on a large CL takes a long time, so you will have lots of conflicts when you merge, and you will have to merge frequently.
- Easier to design well. It’s a lot easier to polish the design and code health of a small change than it is to refine all the details of a large change.
- Less blocking on reviews. Sending self-contained portions of your overall change allows you to continue coding while you wait for your current CL in review.
- Simpler to roll back. A large CL will more likely touch files that get updated between the initial CL submission and a rollback CL, complicating the rollback (the intermediate CLs will probably need to be rolled back too).
- Note that reviewers have discretion to reject your change outright for the sole reason of it being too large. Usually they will thank you for your contribution but request that you somehow make it into a series of smaller changes. It can be a lot of work to split up a change after you’ve already written it, or require lots of time arguing about why the reviewer should accept your large change. It’s easier to just write small CLs in the first place.
```
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
