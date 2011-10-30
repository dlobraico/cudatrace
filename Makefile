obj = cudatrace.o
bin = cudatrace

CC = nvcc
CFLAGS = -O3 

$(bin): $(obj)
	$(CC) -o $@ $(obj) -lm -lpthread

.PHONY: clean
clean:
	rm -f $(obj) $(bin)

.PHONY: install
install:
	cp $(bin) /usr/local/bin/$(bin)

.PHONY: uninstall
uninstall:
	rm -f /usr/local/bin/$(bin)
