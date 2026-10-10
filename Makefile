# roses.cpp — multithreaded C++20 core (stdlib only)
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra
LDFLAGS  ?= -pthread

roses: roses.cpp
	$(CXX) $(CXXFLAGS) -o roses roses.cpp $(LDFLAGS)

# thread-sanitizer build for race checking (slow; not for play)
roses-tsan: roses.cpp
	$(CXX) $(CXXFLAGS) -O1 -fsanitize=thread -o roses-tsan roses.cpp $(LDFLAGS)

clean:
	rm -f roses roses-tsan

.PHONY: clean
