![Logo]({{tool.genice.urls.logo}})

# GenIce-core

{{tool.poetry.description}}

version {{tool.poetry.version}}

## Requirements

{% for item in tool.poetry.dependencies %}\* {{item}}
{% endfor %}

## Installation

GenIce-core is registered to [PyPI (Python Package Index)]({{tool.genice.urls.repository}}).
Install with pip3.

    pip3 install genice-core

## Uninstallation

    pip3 uninstall genice-core

## API

API manual is [here]({{project.urls.manual}}).

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

GenIce has been available as open source software on GitHub({{tool.genice.urls.repository}}) since 2015. Feedback, suggestions for improvements and enhancements, bug fixes, etc. are sincerely welcome. Developers and test users are also welcome. If you have any ice that is publicly available but not included in GenIce, please let us know.
