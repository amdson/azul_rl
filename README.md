# azul_rl

### Quick Start

Open the notebook *final_hw.ipynb* and follow the instructions included in there. The first cell
will install needed dependencies. If you wish to install them separately, you can install the
following list of packages: *flax*, *jax*, *optax*, *networkk*, and *colorama*.

### File Summary

The files are as follows:

**Azul_Simulator.py**: Code for simulating playthroughs of the game of Azul, and for interacting
with the simulator via a terminal (includes colorized output). 

**Azul_Visuals.py**: Code for visualizing the Azul gameboard using *matplotlib*.

**final_hw.ipynb**: An iPython notebook containing the actual homework assignment. Code blocks to fill
in are marked with TODOs, and written questions appear in Markdown cells. Broadly, it covers:

1. Implementation of the machine learning approach outlined in the paper using Jax/Flax.

2. Implementation and exploration of the Monte Carlo tree search algorithm.

3. Model training and evaluation for the game of tic-tac-toe.

4. Model training and evaluation for the game of Azul.

A PDF version is included as well.

**final_hw_solutions.ipynb**: An iPython notebook containing solutions to the homework. A PDF version
is includes as well.

### Sources

[1](https://kstatic.googleusercontent.com/files/2f51b2a749a284c2e2dfa13911da965f4855092a179469aedd15fbe4efe8f8cbf9c515ef83ac03a6515fa990e6f85fd827dcd477845e806f23a17845072dc7bd?fbclid=IwAR1CiRCE0a5nrZBQs2A2Ezw3fh3VUg7JWFC0m8ZKNDIp4xOzqPuhUmTgYQk)
Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A
general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science,
362(6419), 1140-1144.
