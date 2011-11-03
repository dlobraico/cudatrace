obj = cudatrace.o
bin = cudatrace
src = cudatrace.cu

CC = nvcc
CFLAGS = -g -G -O0 -arch sm_23 -lm -lpthread

$(bin): $(src)
	$(CC) -o $@ $(src) $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) $(bin)

.PHONY: install
install:
	cp $(bin) /usr/local/bin/$(bin)

.PHONY: uninstall
uninstall:
	rm -f /usr/local/bin/$(bin)
