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

### Keijzer 8
The `Keijzer 8` function is given by the formula:

$y = \log(x)$

Note that this function is not defined for $x <= 0$

#### Suggested Ranges
- Range 1: $(.05,20)$


#### Plots
- Range 1, $(.05,20)$:

![Keijzer 8 Range 1](keijzer8-.0520.png)

---

### Keijzer 9
The `Keijzer 9` function is given by the formula:

$y = \sqrt{x}$

Note that this function is not defined for $x < 0$

#### Suggested Ranges
- Range 1: $(0,1000)$


#### Plots
- Range 1, $(0,1000)$:

![Keijzer 9 Range 1](keijzer9-1000.png)

---

### Keijzer 10
The `Keijzer 10` function is given by the formula:

$y = \arcsin(x)$


#### Suggested Ranges
- Range 1: $(-10,10)$


#### Plots
- Range 1, $(-10,10)$:

![Keijzer 10 Range 1](keijzer10-1010.png)

---

### Keijzer 11
The `Keijzer 11` function is given by the formula:

$z = x^y$


#### Suggested Ranges
- Range 1: $(0,2)$


#### Plots
- Range 1, $(0,2)$:

![Keijzer 11 Range 1](keijzer11-02.png)

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

---

### Keijzer 13
The `Keijzer 13` function is given by the formula:

$z = x^4 - x^3 + \frac{y^2}{2} - y$


#### Suggested Ranges
- Range 1: $(-1,1)$


#### Plots
- Range 1, $(-1,1)$:

![Keijzer 13 Range 1](keijzer13-11.png)

---

### Keijzer 14
The `Keijzer 14` function is given by the formula:

$z = 6 \sin(x) \cos(y)$

#### Suggested Ranges
- Range 1: $(-\pi, \pi)$
- Range 2: $(-2\pi, 2\pi)$

#### Plots
- Range 1, $(-\pi, \pi)$:

![Keijzer 14 Range 1](keijzer14-pipi.png)
- Range 2, $(-2\pi, 2\pi)$:

![Keijzer 14 Range 2](keijzer14-2pi2pi.png)

---

### Keijzer 15
The `Keijzer 15` function is given by the formula:

$z = \frac{8}{2 + x^2 + y^2}$


#### Suggested Ranges
- Range 1: $(-2,2)$


#### Plots
- Range 1, $(-2,2)$:

![Keijzer 15 Range 1](keijzer15-22.png)

---

### Keijzer 16
The `Keijzer 16` function is given by the formula:

$z = \frac{x^3}{5} + \frac{y^3}{2} - y - x$

#### Suggested Ranges
- Range 1: $(-1, 1)$
- Range 2: $(-2, 2)$

#### Plots
- Range 1, $(-1, 1)$:

![Keijzer 16 Range 1](keijzer16-11.png)
- Range 2, $(-2, 2)$:

![Keijzer 16 Range 2](keijzer16-22.png)

---

### Linobi 0
The `Linobi 0` function is given by the formula:

$y = \sin(x)$


#### Suggested Ranges
- Range 1: $(-\pi,\pi)$


#### Plots
- Range 1, $(-\pi,\pi)$:

![Linobi 0 Range 1](linobi0-pipi.png)

---

### Linobi 1
The `Linobi 1` function is given by the formula:

$z = \sin(\sqrt{x^2 + y^2})$


#### Suggested Ranges
- Range 1: $(-5,5)$


#### Plots
- Range 1, $(-5,5)$:

![Linobi 1 Range 1](linobi1-55.png)

---

### Linobi 2
The `Linobi 2` function is given by the formula:

$z = x^2 - y^2 + y - 1$


#### Suggested Ranges
- Range 1: $(-1,1)$
- Range 2: $(-5,5)$


#### Plots
- Range 1, $(-5,5)$:

![Linobi 2 Range 1](linobi2-11.png)
- Range 2, $(-5,5)$:

![Linobi 2 Range 2](linobi2-55.png)

---

### Linobi 3
The `Linobi 3` function is given by the formula:

$z = \frac{\sin(5x(3y+1))+1}{2}$


#### Suggested Ranges
- Range 1: $(-.5,.5)$
- Range 2: $(-1,1)$
- Range 2: $(-1.5,1.5)$


#### Plots
- Range 1, $(-.5,.5)$:

![Linobi 3 Range 1](linobi3-0505.png)
- Range 2, $(-1,1)$:

![Linobi 3 Range 2](linobi3-11.png)
- Range 3, $(-1.5,1.5)$:

![Linobi 3 Range 3](linobi3-1515.png)
