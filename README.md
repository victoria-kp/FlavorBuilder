# FlavorBuilder
A unified Python framework for model building with finite flavor symmetries in particle physics. It integrates group-theory computations via [GAP](https://www.gap-system.org/) with model-building tools for constructing and analyzing flavor models. This package serves as a core component supporting our work in [arXiv:2506.08080 [hep-ph]](https://arxiv.org/pdf/2506.08080).

**PyDiscrete**: a Python reimplementation of the Mathematica package [Discrete](https://discrete.hepforge.org/), enabling group-theory computations via [GAP](https://www.gap-system.org/). It extracts group properties, character tables, computes Kronecker products and Clebsch-Gordan coefficients. See the tutorial notebook "TutorialPyDiscrete.ipynb" for more details.

**Model2Mass**: a symbolic and numerical framework for constructing Lagrangians, derive mass matrices, and compute phenomenologically relevant quantities for BSM flavor models with finite flavor symmetry.

The Tutorials folder contains Jupyter notebooks demonstrating how to use the two core packages. The FastEvaluation folder includes a notebook with all the necessary code to derive the analytical expressions for the mass matrices and compute the corresponding χ² values using the parameters stored in the .txt files. This notebook can also be used to validate the neutrino flavor models presented in [arXiv:2506.08080 [hep-ph]](https://arxiv.org/pdf/2506.08080) .

# Requirements:
1. Python ≥ 3.9
2. [GAP](https://www.gap-system.org/) ≥ 4.11.1
3. Dependencies: numpy, sympy, pandas, json, os, itertools, random, collections, numbers, subprocess, ast, re, and the optionally IPython (for LaTeX table visualization)

# Installation 
Option 1: You can install FlavorBuilder from PyPI with pip by running
```bash

   pip install FlavorBuilder
```

Option 2: You can also download the package from [GitHub](https://github.com/victoria-kp/PyDiscrete_And_Model2Mass) and add it to your Python path:

```bash

   import sys, os
   src_path = "path/FlavorBuilder/src"
   sys.path.insert(0, os.path.abspath(src_path))

   from FlavorBuilder.PyDiscrete import Group
   from FlavorBuilder.Model2Mass import Model2Mass
```



# References and Citations

If you use FlavorBuilder in academic work, please cite the following:
1. GAP library: The GAP Group. GAP – Groups, Algorithms, and Programming, Version 4.14.0, [https://www.gap-system.org](https://www.gap-system.org) (2024)
2. [Discrete](https://discrete.hepforge.org/): Mathematica Discrete by Martin Holthausen, and Michael A. Schmidt [arXiv:1111.1730](http://arxiv.org/abs/1111.1730)
3. [Flavorpy](https://github.com/FlavorPy/FlavorPy/tree/master): [A. Baur, "FlavorPy", Zenodo, 2024, doi: 10.5281/zenodo.11060597](https://zenodo.org/records/12191928)
4. If you use the NuFit experimental data in FlavorPy, please also cite:
I. Esteban, M. C. González-García, M. Maltoni, T. Schwetz, and A. Zhou, The fate of hints: updated global analysis of three-flavor neutrino oscillations, JHEP 09 (2020), 178, [arXiv:2007.14792](https://arxiv.org/abs/2007.14792) [hep-ph], https://www.nu-fit.org.
5. Our work: J. B. Baretz, M. Fieg, V. Ganesh, A. Ghosh, V. Knapp-Perez, J. Rudolph and D. Whiteson, "Towards AI-assisted Neutrino Flavor Theory Design", [arXiv:2506.08080 [hep-ph]](https://arxiv.org/pdf/2506.08080)

# Credit
PyDiscrete is partly the translation of [Discrete](https://discrete.hepforge.org/) 

# Contact and Maintenance
The FlavorBuilder software is authored by Victoria Knapp Pérez and Jake Rudolph. For questions, bug reports, or collaboration inquiries, please contact Victoria by [email](vknapppe@uci.edu) or open an issue on the [GitHub](https://github.com/victoria-kp/PyDiscrete_And_Model2Mass) repository.



