.PHONY: all clean

CC=gcc
LN=gcc

all: train test
# type make/make all to compile test_hmm

train: train.o hmm.o
	$(LN) -o train train.o hmm.o

test: test.o hmm.o
	$(LN) -o test test.o hmm.o

%.o: %.c
	$(CC) -c $<

clean:
	$(RM) train test train.o test.o hmm.o # type make clean to remove the compiled file
