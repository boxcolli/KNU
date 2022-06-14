a.out : classes.o main.o
	g++ -o a.out classes.o main.o

classes.o : classes.cpp
	g++ -c -o classes.o classes.cpp

main.o : main.cpp
	g++ -c -o main.o main.cpp

clean:
	rm *.o a.out