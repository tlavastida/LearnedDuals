# LearnedDuals

Code for the paper "Faster Matching via Learned Duals" in NeurIPS 2021.  
Authors: Michael Dinitz, Sungjin Im, Thomas Lavatida, Benjamin Moseley, and Sergeri Vassilvitskii  
[Openreview Link](https://openreview.net/forum?id=kB8eks2Edt8)

Code is provided as is.

## Overview

The code files can be roughly split into two groups: algorithm implementations + utility code and code for running the experiments.

### Implementation + Utility Code Files


| File name | Description |
| ----------- | ----------- |
| BipartiteGraph.py | Bipartite graph data structure |
| MaxBipartiteMatching.py | Maximum cardinality matching in bipartite graphs via Hopcroft-Karp |
| MinWeightPerfectMatching.py | Minimum weight perfect matching in bipartite graphs via the Hungarian algorithm.  Also supports initialization with user-provided dual variables. |
| UtilityFunctions.py | Utility code that is used in several places |
| InstanceGeneration.py | Utility code for generating random instances for testing |


### Code for Running Experiments

| File name | Description |
| ----------- | ----------- |


### Dependencies

- Numpy