CXX ?= g++
MPICXX ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp
BUILD := build

.PHONY: all clean test
all: $(BUILD)/heat_serial $(BUILD)/heat_parallel

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/heat_serial: src/serial.cpp src/common.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ src/serial.cpp

$(BUILD)/heat_parallel: src/parallel.cpp src/common.hpp | $(BUILD)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ src/parallel.cpp

test: all
	python3 scripts/validate.py

clean:
	rm -rf $(BUILD)
