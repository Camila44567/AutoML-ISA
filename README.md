# AutoML-ISA
My master's project on automatically extracting intervals of easy or hard instances for creating hardness rules.

<!--
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/ita-ml%2Finstance-hardness/binder?filepath=notebooks%2F)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://en.wikipedia.org/wiki/MIT_License)
-->

# PyHard

_Instance Hardness analysis in Python_

<!--![picture](docs/img/circle-fs.png)-->

## Getting Started

PyHard employes a methodology known as [_Instance Space Analysis_](https://github.com/andremun/InstanceSpace) (ISA) to analyse performance at the instance level rather than at dataset level. The result is an alternative for visualizing algorithm performance for each instance within a dataset, by relating predictive performance to estimated instance hardness measures extracted from the data. This analysis reveals regions of strengths and weaknesses of predictors (aka _footprints_), and highlights individual instances within a  dataset that warrant further investigation, either due to their unique properties or potential data quality issues.

By default, the following steps shall be taken:

1. Calculate the _hardness measures_;

2. Evaluate classification performance at instance level for each algorithm;

3. Select the most relevant hardness measures with respect to the instance classification error;

4. Join the outputs of steps 1, 2 and 3 to build the _metadata_ file (`metadata.csv`);

5. Run __ISA__ (_Instance Space Analysis_), which generates the _Instance Space_ (IS) representation and the _footprint_ areas;

6. To explore the results from step 5, launch the visualization dashboard:  
``pyhard --app``


### Input file

Please follow the guidelines below:

* Only `csv` files are accepted

* The dataset should be in the format `(n_instances, n_features)`

* **Do not** include any index column. Instances will be indexed in order, starting from **1**

* **The last column** must contain the classes of the instances

* Categorical features should be handled previously

## References

_Base_

1. Michael R. Smith, Tony Martinez, and Christophe Giraud-Carrier. 2014. __An instance level analysis of data complexity__. Mach. Learn. 95, 2 (May 2014), 225–256.

Luengo J, Herrera F. _An  automatic  extraction  method  of  the  domains  of  competence  for learning classifiers using data complexity measures_. Knowledge and Information Systems. 2013 Oct;42(1):147–180. Available from:https://doi.org/10.1007/s10115-013-0700-4.
