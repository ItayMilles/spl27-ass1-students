all: bin/server

bin/server: bin/Main.o bin/Server.o bin/Model.o
	g++ -o bin/server bin/Main.o bin/Server.o bin/Model.o

bin/Main.o: src/Main.cpp
	g++ -Wall -Weffc++ -std=c++11 -g -c -Iinclude src/Main.cpp -o bin/Main.o

bin/Server.o: src/Server.cpp include/Server.h
	g++ -Wall -Weffc++ -std=c++11 -g -c -Iinclude src/Server.cpp -o bin/Server.o

bin/Model.o: src/Model.cpp include/Model.h
	g++ -Wall -Weffc++ -std=c++11 -g -c -Iinclude src/Model.cpp -o bin/Model.o

clean:
	rm -f bin/*.o bin/server