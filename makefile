julia: dirs object/julia.o object/utils.o object/qdbmp.o
	nvcc -L. -Iinclude  object/qdbmp.o  object/utils.o  object/julia.o -o julia

object/julia.o: 
	nvcc -c -L. -Iinclude src/julia.cu -o object/julia.o -lm

object/qdbmp.o:
	gcc -c -fopenmp -Wall --std=c99 -L. -Iinclude src/qdbmp.c -o object/qdbmp.o -lm

object/utils.o:
	nvcc -c -L. -Iinclude src/utils.cu -o object/utils.o -lm

object/julia-c.o: include/julia-c.h
	gcc -c -fopenmp -Wall --std=c99 -L. -Iinclude src/julia-c.c -o object/julia-c.o -lm

clean:
	rm -f output/bmpout/* &
	rm -f output/julia.mkv &
	rm -f object/* &
	rm julia &

build-and-run: clean julia
	clear
	./julia

dirs:
	mkdir object &
	mkdir output && mkdir output/bpmout &
	mkdir output/bmpout &

julia.mkv: build-and-run
	ffmpeg -i 'output/bmpout/%05d_out.bmp' -r 60 output/julia.mkv