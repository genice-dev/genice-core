![Logo](https://raw.githubusercontent.com/vitroid/GenIce/develop/logo/genice-v0.png)

# GenIce-core

Core algorithms of GenIce2 It provides algorithms to generate directed graphs that satisfy the ice rules (hydrogen-disordered ice) from an undirected graph.

version 1.2.1

## Requirements

Python 3.10 or newer is required (see `pyproject.toml`).

- python
- networkx
- numpy


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

Additional examples can be found at https://github.com/vitroid/genice-core

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

GenIce has been available as open source software on [GitHub](https://github.com/genice-dev/genice-core) since 2015. Feedback, suggestions for improvements and enhancements, bug fixes, etc., are sincerely welcome. Developers and test users are also welcome. If you have any publicly available ice that is not included in GenIce, please let us know.

## Development

Install dependencies (including dev and test groups):

    poetry install

Run tests:

    poetry run pytest

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
