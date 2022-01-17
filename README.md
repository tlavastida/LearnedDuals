# LearnedDuals

Code for the paper "Faster Matching via Learned Duals" which appeared in NeurIPS 2021.  
Authors: Michael Dinitz, Sungjin Im, Thomas Lavatida, Benjamin Moseley, and Sergei Vassilvitskii  
[Openreview Link](https://openreview.net/forum?id=kB8eks2Edt8)

Code is provided as is.

See Section 4 of [the paper](https://openreview.net/pdf?id=kB8eks2Edt8) for details on the experiments.  Datasets can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

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
| type_model_exp.py | For synthetic experiments in the batch setting on the type model instances |
| geometric_type_model_exp.py | For experiments in the batch setting on the instances derived from geometric data  sets |
| online_test.py | For experiments in the online setting (both synthetic and real data sets) |

### Dependencies

- Numpy
