
setup:
	python setup.py

nuke:
	rm -f .DS* ._* ._.* ./src/.DS* \
	./src/._* ./src/._.* ./src/*.pyc \
	./tests/.DS* ./tests/._* ./tests/._.* ./tests/*.pyc
