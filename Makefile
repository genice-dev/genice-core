SOURCES=$(wildcard genice_core/*.py)

all: README.md
	echo Hello.

# https://qiita.com/yukinarit/items/0996180032c077443efb
# https://zenn.dev/atu4403/articles/python-githubpages
doc: README.md CITATION.cff 
	pdoc -o docs ./genice_core --docformat google

test-deploy:
	poetry publish --build -r testpypi
test-install:
	pip install --index-url https://test.pypi.org/simple/ genice-core


uninstall:
	-pip uninstall -y genice-core


deploy:
	poetry publish --build


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
