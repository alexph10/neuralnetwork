CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall
TARGET := neural_net

SRC := main.cpp tensor.cpp layer.cpp model.cpp
OBJ := $(SRC:.cpp=.o)
LIBS :=

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
