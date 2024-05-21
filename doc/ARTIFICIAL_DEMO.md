## Standard Artificial Regression Problems Collection

## Overview
TurboGP comes with a comprehensive set of artificial regression problems that are ready to be tested with the library. Most of these problems are classical benchmarks commonly found in the research literature, but there are also some unique problems that have not been previously proposed within the GP literature. Each problem is detailed with its mathematical formula, suggested ranges for generating the function, and illustrative plots. For proper references and further details, you can refer to the source file that generates these datasets, located at `Utils/ArtificialRegDataset.py`.




### Keijzer 3
The `Keijzer 12` function is given by the formula:

$y = 3.1416x^2 + 100$

#### Suggested Ranges
- Range 1: $(-10, 10)$


#### Plots
- Range 1, $(-10, 10)$:

![Keijzer 3 Range 1](keijzer3.png)



### Keijzer 12
The `Keijzer 12` function is given by the formula:

$z = (xy) + \sin((x - 1)(y - 1))$

#### Suggested Ranges
- Range 1: $(-0.5\pi, 0.5\pi)$
- Range 2: $(-\pi, \pi)$
- Range 3: $(10, 10)$

#### Plots
- Range 1, $(-0.5\pi, 0.5\pi)$:

![Keijzer 12 Range 1](keijzer12-05pi05pi.png)
- Range 2, $(-\pi, \pi)$:

![Keijzer 12 Range 2](keijzer12-pipi.png)
- Range 3, $(10, 10)$:

![Keijzer 12 Range 3](keijzer12-1010.png)




