obj = cudatrace.o
bin = cudatrace
src = cudatrace.cu

CC = nvcc
CFLAGS = -g -G -O0 -arch sm_13 -lm -lpthread

$(bin): $(src)
	$(CC) -o $@ $(src) $(CFLAGS)

.PHONY: test
test:
	./$(bin) -i c-ray-1.1/scene -o scene.jpg

.PHONY: clean
clean:
	rm -f $(obj) $(bin)

.PHONY: install
install:
	cp $(bin) /usr/local/bin/$(bin)

.PHONY: uninstall
uninstall:
	rm -f /usr/local/bin/$(bin)
