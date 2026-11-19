# gpuseed — build
#   make                 # build for Ada (sm_89)
#   make ARCH=sm_120     # build for Blackwell (5090)
#   make selftest        # build + run the correctness self-test
#   make clean

NVCC   ?= nvcc
ARCH   ?= sm_89
CFLAGS  = -O3 -arch=$(ARCH) -Xcompiler -fopenmp
LIBS    = -lcrypto -lgomp

gpuseed: gpuseed.cu known_keys.h hitcrypt.h old_words.txt
	$(NVCC) $(CFLAGS) gpuseed.cu -o gpuseed $(LIBS)

selftest: gpuseed
	./gpuseed --selftest

clean:
	rm -f gpuseed

.PHONY: selftest clean
