target = csampling.so

all: $(target)

$(target): csampling.c
	gcc -O3 -shared -Wl,-soname,$@ -o $@ -fPIC $< -llapack -llapacke

clean:
	rm $(target)
