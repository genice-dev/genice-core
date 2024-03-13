![Logo](https://raw.githubusercontent.com/vitroid/GenIce/develop/logo/genice-v0.png)

# GenIce-core

Core algorithms of GenIce2

version 0.9


## Requirements

* python
* networkx
* numpy
* pdoc3



## Installation

GenIce-core is registered to [PyPI (Python Package Index)](https://pypi.python.org/pypi/GenIce).
Install with pip3.

    pip3 install genice-core

## Uninstallation

    pip3 uninstall genice-core

## API

API manual is [here](https://genice-dev.github.io/genice-core).

## Examples

Make an ice graph from a given undirected graph.

```python
import networkx as nx
import matplotlib
import genice_core

# np.random.seed(12345)

g = nx.dodecahedral_graph()  # dodecahedral 20mer
pos = nx.spring_layout(g)

# set orientations of the hydrogen bonds.
dg = genice_core.ice_graph(g)

nx.draw_networkx(dg, pos)
```


## Algorithms and how to cite them.

The algorithms to make a depolarized hydrogen-disordered ice are explained in these papers:

M. Matsumoto, T. Yagasaki, and H. Tanaka,"GenIce: Hydrogen-Disordered
Ice Generator",  J. Comput. Chem. 39, 61-64 (2017). [DOI: 10.1002/jcc.25077](http://doi.org/10.1002/jcc.25077)

    @article{Matsumoto:2017bk,
        author = {Matsumoto, Masakazu and Yagasaki, Takuma and Tanaka, Hideki},
        title = {GenIce: Hydrogen-Disordered Ice Generator},
        journal = {Journal of Computational Chemistry},
		volume = {39},
		pages = {61-64},
        year = {2017}
    }

M. Matsumoto, T. Yagasaki, and H. Tanaka, “GenIce-core: Efficient algorithm for generation of hydrogen-disordered ice structures.”, J. Chem. Phys. 160, 094101 (2024). [DOI:10.1063/5.0198056](https://doi.org/10.1063/5.0198056)

    @article{Matsumoto:2024,
        author = {Matsumoto, Masakazu and Yagasaki, Takuma and Tanaka, Hideki},
        title = {GenIce-core: Efficient algorithm for generation of hydrogen-disordered ice structures},
        journal = {Journal of Chemical Physics},
        volume = {160},
        pages = {094101},
        year = {2024}
    }

## How to contribute

GenIce has been available as open source software on GitHub(https://github.com/vitroid/GenIce) since 2015. Feedback, suggestions for improvements and enhancements, bug fixes, etc. are sincerely welcome. Developers and test users are also welcome. If you have any ice that is publicly available but not included in GenIce, please let us know.
