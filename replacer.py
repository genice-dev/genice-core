import jinja2 as jj
import toml
import sys
import genice_core

proj = toml.load("pyproject.toml")
proj |= {
    "version": genice_core.__version__,
}

t = jj.Environment(loader=jj.FileSystemLoader(".")).from_string(sys.stdin.read())
print(t.render(**proj))
