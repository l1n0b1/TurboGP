## Standard Artificial Regression Problems Collection

## Overview
TurboGP comes with a comprehensive set of artificial regression problems that are ready to be tested with the library. Most of these problems are classical benchmarks commonly found in the research literature, but there are also some unique problems that have not been previously proposed within the GP literature. Each problem is detailed with its mathematical formula, suggested ranges for generating the function, and illustrative plots. For proper references and further details, you can refer to the source file that generates these datasets, located at `Utils/ArtificialRegDataset.py`.


---

### Keijzer 3
The `Keijzer 3` function is given by the formula:

$y = 3.1416x^2 + 100$

#### Suggested Ranges
- Range 1: $(-10, 10)$


#### Plots
- Range 1, $(-10, 10)$:

![Keijzer 3 Range 1](keijzer3.png)

---

### Keijzer 4
The `Keijzer 4` function is given by the formula:

$y = 0.3 x \sin(2 \pi x)$

#### Suggested Ranges
- Range 1: $(-\pi, \pi)$
- Range 2: $(10, 10)$


#### Plots
- Range 1, $(-\pi, \pi)$:

![Keijzer 4 Range 1](keijzer4-3.14.png)
- Range 2, $(10, 10)$:

![Keijzer 4 Range 2](keijzer4-10.png)

---

### Keijzer 5
The `Keijzer 5` function is given by the formula:

$y = x^3 e^{-x} \cos(x) \sin(x) (\sin(x)^2 \cos(x) - 1)$

#### Suggested Ranges
- Range 1: $(0, 2\pi)$
- Range 2: $(-.5\pi, 2\pi)$
- Range 3: $(-2\pi, 2\pi)$

#### Plots
- Range 1, $(0, 2\pi)$:

![Keijzer 5 Range 1](keijzer5-02pi.png)
- Range 2, $(-.5\pi, 2\pi)$:

![Keijzer 5 Range 2](keijzer5-.5pi2pi.png)
- Range 3, $(-2\pi, 2\pi)$:

![Keijzer 5 Range 3](keijzer5-2pi2pi.png)

---

### Keijzer 7
The `Keijzer 7` function is given by the formula:

$y = {{\sum_{i}}^{\lfloor x \rfloor - 1}} \frac{1}{i}$

#### Suggested Ranges
- Range 1: $(1,30)$
- Range 2: $(1,20)$
- Range 3: $(1,10)$

Note that this function is not defined for $x < 1$

#### Plots
- Range 1, $(1,30)$:

![Keijzer 7 Range 1](keijzer7-30.png)
- Range 2, $(1,20)$:

![Keijzer 7 Range 2](keijzer7-20.png)
- Range 3, $(1,10)$:

![Keijzer 7 Range 3](keijzer7-10.png)

---

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




