CC = gcc
PYCC = cython
PYHEADERS = /usr/include/python2.7

PYFLAGS = -v -2
CFLAGS = -shared -pthread -fPIC -fwrapv -O2 -fno-strict-aliasing -I$(PYHEADERS)

PYXSRC = hidden.pyx
CSRC = hidden.c
MODULE = hidden.so

all: $(CSRC) $(MODULE)

$(CSRC): $(PYXSRC)
	$(PYCC) $(PYFLAGS) $^

$(MODULE): $(CSRC)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm $(MODULE)

