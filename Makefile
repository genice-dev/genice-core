SOURCES=$(wildcard genice_core/*.py)

all: README.md
	echo Hello.

build: $(SOURCES) README.md doc
	python3 -m build

# https://qiita.com/yukinarit/items/0996180032c077443efb
# https://zenn.dev/atu4403/articles/python-githubpages
doc: README.md CITATION.cff 
	pdoc -o docs ./genice_core --docformat google



# pep8:
# 	autopep8 -r -a -a -i clustice/
test-deploy: build
	-pip install twine
	twine upload -r pypitest dist/*
test-install:
	pip install --index-url https://test.pypi.org/simple/ genice-core


# install:
# 	./setup.py install
uninstall:
	-pip uninstall -y genice-core


deploy: build
	twine upload dist/*
# check:
# 	./setup.py check


%: %.j2 replacer.py pyproject.toml
	python replacer.py < $< > $@



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
