# Testing

## Background
Bystro is an instrument for scientific and clinical research
pertaining to human health.  As such, it's important to be able to
write software that is both (1) correct and (2) demonstrates
assurances of its correctness.  While there is in general no silver
bullet for assuring the correctness of complex programs, software
testing is a widely-adopted strategy for increasing software
reliability, affording:

- proofs of correctness for example input-output pairs or even entire properties of functions
- protection against regressions
- executable documentation
- human-readable examples of API usage
- stronger incentives for effective design
- faster and more effective code reviews
- increased velocity in debugging [^1]

## Psychological Barriers to Effective Testing
Testing is often perceived to be a laborious chore, a consideration
that can be postponed until after the real work of developing
application code is complete.  This sentiment is sometimes reinforced
by the notion that, as intelligent people, we can simply see for
ourselves that our code is correct without need for testing.  This
sentiment is undercut by the following observations:

- It is often not complex algorithmic errors that lead to software
  faults, but simple errors like typos and "thinkos" that lead to
  the most pernicious software faults.
- Even if we can verify the correctness of our own code, we can't
  verify that it *remains* correct whenever code is modified.  Those
  modifications may come from the module under test, another module
  within Bystro, or a library outside of Bystro entirely.
- Software, like any complex system, is subject to *emergence*, where
  the system exhibits behaviors at one level of organization that are
  essentially unpredictable in principle from analysis of the levels
  below.  At some level of complexity, the only practical assurance of
  correctness is empirical testing.

Reluctance to perform testing can lead to the following organizational
equilibrium:

- low test coverage
- a test suite that doesn't afford much benefit
- code that is difficult to test
- unfamiliarity with efficient testing workflows leading to high
  perceived costs of writing tests
- psychological reluctance / unease with working on the codebase
  
But there is another equilibrium where the following
conditions hold:

- high test coverage
- a test suite that assures many software properties
- code that's easy to test
- familiarity with efficient testing workflows, leading low perceived
  cost of testing and interactive development
- psychological confidence for modifying code

The latter state is generally preferable to the former, especially for
long-running projects where multiple authors will collaborate to
produce multiple versions of a program over time.


## How Should We Write Tests?
When considering what to test, the overriding principle is the
so-called [Beyonce
Rule](https://abseil.io/resources/swe-book/html/ch11.html#the_beyonceacutesemicolon_rule):
"if you liked it, then you should have put a test on it".  If it's
important that your code exhibit a certain behavior or property,
ensure it with an automated test.  If not, assume that it will
eventually be broken.

### General Considerations
In general, it should always be possible to run unit tests locally so
that they are run early and often during development.  This means that
unit tests should be:

- self-contained, requiring no connections to the outside world
- deterministic, giving the same result each time
- fast enough that running tests doesn't force a context switch

Tactical recommendations for ensuring these criteria are given below.

### Code coverage 
Many organizations use numerical metrics, such as the fraction of
lines of code in a module executed during tests, in order to set
standards for testing.  There is a certain logic to this: if code is
not under test at all, then it is not tested.  The converse, however,
is not as reliable.  If a given module has 100% test coverage, we can
have assurance that the code doesn't immediately crash upon execution,
and that is better than no assurance at all.  That, however is no
guarantee that the tests effectively verify any property of the
program more complex than "not crashing".

Code coverage reports can be a helpful tool for finding untested
behaviors in our code, and well-tested systems will tend to have high
test coverage, but a system with high test coverage is not necessarily
well-tested.  Worse, making code coverage a target [tends to
encourage](https://en.wikipedia.org/wiki/Goodhart's_law) developers to
write trivial tests to satisfy coverage requirements as easily as
possible.


## Unit Testing in Python

Bystro uses the [pytest](https://docs.pytest.org/en/7.4.x/) framework
for testing Python code.  While the reader is referred to the pytest
documentation for a full discussion of its use, we note the following
points in particular.

### Slow tests
Sometimes you may find that a test is slow, where "slow" means "so
slow that it interrupts you."

First, ask yourself: "is it necessary that this test is slow?"  Slow
tests are often a sign that either the test or the underlying
application code could be better structured.  Even in a dynamic,
high-level language such as Python, a program can do a lot of work
before it takes enough time for human attention to consciously
register its wall time.  If a test is taking a noticeably long time,
consider whether it's doing things that shouldn't be done during a
unit test, like talking to the outside world or doing more work than
necessary in order to verify the property under test.

If a test is irreducibly slow (say, more than a few seconds), mark it
with pytest as follows:

```python
@pytest.mark.slow
def some_slow_test():
    pass
```

If a test takes more than 30 seconds, it is almost certainly either
not correctly structured or not a unit test, and should be somehow
refactored or reorganized.

In all cases, do not prematurely / speculatively optimize tests before
they present problems.

# Footnotes
[^1]: [Software Engineering at Google:
    Testing](https://abseil.io/resources/swe-book/html/ch11.html)



