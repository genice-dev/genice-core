SOURCES=$(wildcard genice_core/*.py)
PKGNAME=genice-core
CXX ?= c++
CXXFLAGS ?= -O3 -march=native -std=c++17

all: README.md CITATION.cff 
	echo Hello.

# ベンチ用: 最適化ビルド
build-nim:
	cd genice_nim && nim c --nimcache:./nimcache -d:release -o:run_stdin run_stdin.nim
build-cpp: genice_cpp/run_stdin
genice_cpp/run_stdin: genice_cpp/run_stdin.cpp genice_cpp/genice_core.cpp
	$(CXX) $(CXXFLAGS) -o genice_cpp/run_stdin genice_cpp/run_stdin.cpp
# 200^3 ダイヤモンド: Nim の diamond で生成（Python はメモリで落ちるため使わない）
# PairList を兄弟ディレクトリに置いてビルド。例: gitbox/genice-core, gitbox/PairList
build-diamond:
	cd ../PairList && nim c --nimcache:./nimcache -p:nim -o:../genice-core/test/data/diamond ../genice-core/test/data/diamond.nim

test/data/diamond: test/data/diamond.nim
	$(MAKE) build-diamond

test/data/diamond_%.txt: test/data/diamond
	test/data/diamond $* test/data/diamond_$*.txt

.PHONY: build-nim build-cpp build-diamond benchmarks

# https://qiita.com/yukinarit/items/0996180032c077443efb
# https://zenn.dev/atu4403/articles/python-githubpages
doc: README.md CITATION.cff 
	pdoc -o docs ./genice_core --docformat google
%: temp_% replacer.py pyproject.toml
	python replacer.py < $< > $@

# time -l: 実経過・CPU・最大RSS（macOS）。Linux なら time -v で同様の max RSS が出る。
benchmark%: test/data/diamond_%.txt build-nim build-cpp 
	@echo "--- Nim ---" && /usr/bin/time -l genice_nim/run_stdin < $< > /dev/null
	@echo "--- C++ ---" && /usr/bin/time -l genice_cpp/run_stdin < $< > /dev/null
	@echo "--- Julia ---" && /usr/bin/time -l julia genice_julia/run_stdin.jl < $< > /dev/null
	@if [ $* -le 50 ]; then echo "--- Python ---" && /usr/bin/time -l python run_stdin.py < $< > /dev/null; fi


test-deploy: clean
	poetry publish --build -r testpypi
test-install:
	pip install --index-url https://test.pypi.org/simple/ $(PKGNAME)
uninstall:
	-pip uninstall -y $(PKGNAME)
build: README.md $(wildcard cycles/*.py)
	poetry build
deploy: clean
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
