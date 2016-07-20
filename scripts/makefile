CC = gcc
PYCC = cython
PYLIBS = /usr/include/python2.7

PYFLAGS = -v -2
CFLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I$(PYLIBS)

all: hmm.c hmm.so

hmm.c: hmm.pyx
	$(PYCC) $(PYFLAGS) $^

hmm.so: hmm.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm hmm.so

