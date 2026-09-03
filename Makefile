# Build the sutra tools.
#
#   make vedic   -> the command-line tool (vedic.cpp + vedic_tools.hpp)
#   make check   -> the 41-check suite over the same tool definitions
#   make profile -> the per-sutra capability sweep

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall

all: vedic

vedic: vedic.cpp vedic_tools.hpp vedic_sutras_complete.hpp
	$(CXX) $(CXXFLAGS) -o $@ vedic.cpp

vedic_pde_tool: vedic_pde_tool.cpp vedic_tools.hpp vedic_sutras_complete.hpp
	$(CXX) $(CXXFLAGS) -o $@ vedic_pde_tool.cpp

vedic_capability_profile: vedic_capability_profile.cpp vedic_sutras_complete.hpp
	$(CXX) $(CXXFLAGS) -o $@ vedic_capability_profile.cpp

check: vedic_pde_tool
	./vedic_pde_tool

profile: vedic_capability_profile
	./vedic_capability_profile

clean:
	rm -f vedic vedic_pde_tool vedic_capability_profile

.PHONY: all check profile clean
