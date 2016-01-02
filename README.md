configurator
------------

[![Build Status](https://travis-ci.org/yasserglez/configurator.svg?branch=master)](https://travis-ci.org/yasserglez/configurator)
[![Coverage Status](https://coveralls.io/repos/yasserglez/configurator/badge.svg?branch=master&service=github)](https://coveralls.io/github/yasserglez/configurator?branch=master)

Python package providing different solutions to the problem of
optimizing the user interaction in a configuration process -- as in
minimizing the number of questions asked to a user in order to obtain
a fully-specified product configuration.

An arbitrary product to be configured may be described in terms of two
alternative configuration models. In the first model, the constraints
that regulate the interactions between the aspects of the product that
can be configured are given in the form of if-then rules. In the
second model, the constraints are given as part of a formulation of
the configuration model as a constraint satisfaction problem. The
problem of minimizing the sequence of questions to be presented to the
users is formulated as a Markov decision process. Different solution
methods are implemented: classical dynamic programming algorithms
(such as value iteration), reinforcement learning techniques
(specifically, Q-learning and the Neural Fitted Q-iteration
algorithm), and a genetic algorithm that solves the reinforcement
learning problem as a stochastic optimization problem.

For additional information, please refer to API documentation in the
`docs/` directory and the following publications:

* Saeideh Hamidi, Periklis Andritsos, and Sotirios Liaskos.
  [Constructing adaptive configuration dialogs using crowd data](https://dl.acm.org/citation.cfm?id=2642960).
  In Proceedings of the 29th ACM/IEEE International Conference on Automated Software Engineering, 485-490, 2014.
* Saeideh Hamidi.
  [Automating software customization via crowdsourcing using association rule mining and Markov decision processes](http://yorkspace.library.yorku.ca/xmlui/handle/10315/28216).
  Master's thesis, York University, Ontario, Canada, 2014.
* Yasser Gonzalez-Fernandez.
  [Efficient calculation of optimal configuration processes](http://yorkspace.library.yorku.ca/xmlui/handle/10315/30739).
  Master's thesis, York University, Ontario, Canada, 2015.


Installation
------------

The easiest way to install the package is via the
[Anaconda Python distribution](http://continuum.io/downloads)
(Python 3 is required):

```
conda install -c yasserglez configurator
```


Author
------

Yasser Gonzalez
* Homepage - http://yassergonzalez.com
* Email - contact@yassergonzalez.com


License
-------

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.
