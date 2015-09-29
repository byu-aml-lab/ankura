target = csampling.so
lapacke_headers_location = /usr/include/lapacke

all: $(target)

$(target): csampling.c
	gcc -O3 -shared -Wl,-soname,$@ -o $@ -fPIC -I"$(lapacke_headers_location)" $< -llapack -llapacke

clean:
	rm $(target)
