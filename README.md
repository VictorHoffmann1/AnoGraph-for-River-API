## Implementation of Sketch-Based Anomaly Detection in Streaming Graphs paper
## Overview

This repository contains an implementation, following River conventions, of the methods Anograph and AnoEdgeGlobal, specifically designed for edge anomaly detection. Methods are described in the paper [Sketch-Based Anomaly Detection in Streaming Graphs](https://dl.acm.org/doi/abs/10.1145/3580305.3599504) by BHATIA, Siddharth, WADHWA, Mohit, KAWAGUCHI, Kenji, et al. 

## Usage

### Hcms Class

Higher-order CMS (H-CMS) data structure implementation for anomaly detection.

#### Initialization

```python
from AnoGraph import Hcms

# Create an instance of Hcms with specified parameters
hcms_instance = Hcms(r=2, b=32, d=None)
```

#### Methods

- `hash(elem, i)`: Hash function used to determine the bucket for an element in a specific row.
- `insert(a, b, weight)`: Inserts an edge (a, b) with a specified weight into the count matrix.
- `decay(decay_factor)`: Applies decay to the entire count matrix to decrease counts over time.
- `getAnoEdgeGlobalScore(src, dst)`: Computes the anomaly score for an edge (src, dst).
- `getAnographScore()`: Computes the global anomaly score for the entire count matrix.

### Anograph Class

Anograph class for computing anomaly scores based on matrix density calculations.

#### Initialization

```python
from AnoGraph import Anograph

# Create an instance of Anograph with specified parameters
anograph_instance = Anograph(time_window=30, edge_threshold=50, rows=2, buckets=32)
```

#### Methods

- `learn_one(x)`: Update the Anograph instance with a new edge.
- `get_score()`: Get the current anomaly score.
- `pickMinRow(mat, row_flag, col_flag)`: Pick the row with the minimum sum from the matrix.
- `pickMinCol(mat, row_flag, col_flag)`: Pick the column with the minimum sum from the matrix.
- `getMatrixDensity(mat, row_flag, col_flag)`: Compute the density of the submatrix specified by row_flag and col_flag.
- `getAnographDensity(mat)`: Compute the Anograph density of the matrix.

### AnoEdgeGlobal Class

Anomaly Detection using Edge Global Density.

#### Initialization

```python
from AnoGraph import AnoEdgeGlobal

# Create an instance of AnoEdgeGlobal with specified parameters
ano_edge_global_instance = AnoEdgeGlobal(rows=2, buckets=32, decay_factor=0.9)
```

#### Methods

- `learn_one(x)`: Update the AnoEdgeGlobal instance with a new edge.
- `score_one(x)`: Calculate the anomaly score for a given edge.
- `getAnoEdgeGlobalDensity(mat, src, dst)`: Calculate the density-based anomaly score for a specific edge.

## Dependencies

- Python 3.x