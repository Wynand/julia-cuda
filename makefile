julia: object/julia.o object/utils.o object/qdbmp.o object/julia-c.o
	nvcc -L. -Iinclude  object/julia-c.o  object/qdbmp.o  object/utils.o  object/julia.o -o julia

object/julia.o: 
	nvcc -c -L. -Iinclude src/julia.cu -o object/julia.o -lm

object/qdbmp.o:
	gcc -c -fopenmp -Wall --std=c99 -L. -Iinclude src/qdbmp.c -o object/qdbmp.o -lm

object/utils.o:
	nvcc -c -L. -Iinclude src/utils.cu -o object/utils.o -lm

object/julia-c.o: include/julia-c.h
	gcc -c -fopenmp -Wall --std=c99 -L. -Iinclude src/julia-c.c -o object/julia-c.o -lm

clean:
	rm -f bmpout/* &
	rm -f object/* &
	rm julia &

build-and-run: clean julia
	clear
	./julia

dirs:
	mkdir object &
	mkdir bmpout &