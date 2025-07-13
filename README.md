# **PyNatple: The Description**

This package come up to complete my undergraduate thesis under the Non-linear Algorithms for Advanced Study in near-High-resolution crustal Architecture project, or NATASHA. This package designed to perform Parker-Oldenburg method, a Fourier-transform-based inversion, optimized by genetic algorithm to estimate its hyperprameter precisely. Its original purpose in my thesis is for Moho topographic determination but not limited on it, as mentioned in the original publication by Parker.

The current dependency is still light:
1. NumPy, for numerical based work;
2. xarray, as a main data architecture;
3. xrft, to perform Fourier transform in xarray architecture;
4. That's it!

This package is still juvenile, so there are i vast space to improve and discuss.

Main reference:
1. Oldenburg, D. W. (1974). The inversion and interpretation of gravity anomalies. GEOPHYSICS, 39(4), 526–536. (https://doi.org/10.1190/1.1440444)
2. Parker, R. L. (1973). The Rapid Calculation of Potential Anomalies. Geophysical Journal International, 31(4), 447–455. (https://doi.org/10.1111/j.1365-246X.1973.tb06513.x)
