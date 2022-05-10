a.out : misc.o classes.o main.o
	g++ -o a.out misc.o classes.o main.o

misc.o : MISC.cpp
	g++ -c -o misc.o MISC.cpp

classes.o : CLASSES.cpp
	g++ -c -o classes.o CLASSES.cpp

main.o : MAIN.cpp
	g++ -c -o main.o MAIN.cpp

clean:
	rm *.o a.out