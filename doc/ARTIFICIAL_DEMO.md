## Standard Artificial Regression Problems Collection

## Overview
TurboGP comes with a comprehensive set of artificial regression problems that are ready to be tested with the library. Most of these problems are classical benchmarks commonly found in the research literature, but there are also some unique problems that have not been previously proposed within the GP literature. Each problem is detailed with its mathematical formula, suggested ranges for generating the function, and illustrative plots. For proper references and further details, you can refer to the source file that generates these datasets, located at `Utils/ArtificialRegDataset.py`.




### Keijzer 12
The `Keijzer 12` function is given by the formula:
$$
z = (x \cdot y) + \sin((x - 1) \cdot (y - 1))
$$

#### Suggested Ranges
- Range 1: $(-0.5\pi, 0.5\pi)$
- Range 2: $(-\pi, \pi)$
- Range 3: $(10, 10)$

#### Plots
- Range 1: ![Keijzer 12 Range 1](keijzer12-05pi05pi.png)
- Range 2: ![Keijzer 12 Range 2](keijzer12-pipi.png)
- Range 3: ![Keijzer 12 Range 3](keijzer12-1010.png)
