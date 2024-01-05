SOURCES=$(wildcard genice_core/*.py)
PKGNAME=genice-core

all: README.md
	echo Hello.

# https://qiita.com/yukinarit/items/0996180032c077443efb
# https://zenn.dev/atu4403/articles/python-githubpages
doc: README.md CITATION.cff 
	pdoc -o docs ./genice_core --docformat google
%: temp_% replacer.py pyproject.toml
	python replacer.py < $< > $@

test-deploy:
	poetry publish --build -r testpypi
test-install:
	pip install --index-url https://test.pypi.org/simple/ $(PKGNAME)
uninstall:
	-pip uninstall -y $(PKGNAME)
build: README.md $(wildcard cycles/*.py)
	poetry build
deploy:
	poetry publish --build
check:
	poetry check


clean:
	-rm -rf build dist
distclean:
	-rm *.scad *.yap @*
	-rm -rf build dist
	-rm -rf *.egg-info
	-rm .DS_Store
	find . -name __pycache__ | xargs rm -rf
	find . -name \*.pyc      | xargs rm -rf
	find . -name \*~         | xargs rm -rf
