%%writefile /content/test_vedic.cpp
#include "vedic_complete.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

void test_fixed256() {
    std::cout << "Testing Fixed256..." << std::endl;
    
    vedic::Fixed256 a(100);
    vedic::Fixed256 b(50);
    
    // Addition
    vedic::Fixed256 sum = a + b;
    assert(sum.to_double() == 150.0);
    
    // Subtraction
    vedic::Fixed256 diff = a - b;
    assert(diff.to_double() == 50.0);
    
    // Multiplication
    vedic::Fixed256 prod = a * b;
    assert(std::abs(prod.to_double() - 5000.0) < 1e-6);
    
    // Division
    vedic::Fixed256 quot = a / b;
    assert(std::abs(quot.to_double() - 2.0) < 1e-6);
    
    std::cout << "Fixed256 tests passed!" << std::endl;
}

void test_rational() {
    std::cout << "Testing Rational..." << std::endl;
    
    vedic::Rational a(1, 2);
    vedic::Rational b(1, 3);
    
    // Addition
    vedic::Rational sum = a + b;
    assert(sum.get_numerator() == 5 && sum.get_denominator() == 6);
    
    // Multiplication
    vedic::Rational prod = a * b;
    assert(prod.get_numerator() == 1 && prod.get_denominator() == 6);
    
    // Comparison
    assert(a > b);
    assert(b < a);
    assert(a != b);
    
    std::cout << "Rational tests passed!" << std::endl;
}

void test_sutras() {
    std::cout << "Testing CoreSutras..." << std::endl;
    
    vedic::Rational chi(1);
    
    // S_0(z) = 1
    vedic::Rational s0 = vedic::CoreSutras::S_k(0, chi);
    assert(s0 == vedic::Rational(1));
    
    // S_1(z) = z - 1
    vedic::Rational s1 = vedic::CoreSutras::S_k(1, chi);
    assert(s1 == vedic::Rational(0));
    
    // Derivative: dS_1/dz = 1
    vedic::Rational ds1 = vedic::CoreSutras::dS_k_dz(1, chi);
    assert(ds1 == vedic::Rational(1));
    
    std::cout << "CoreSutras tests passed!" << std::endl;
}

void test_fabric() {
    std::cout << "Testing KroneckerFabric..." << std::endl;
    
    vedic::Rational chi(1);
    vedic::KroneckerFabric fabric(2, chi);
    
    auto matrix = fabric.get_fabric();
    assert(matrix.size() == 4);
    assert(matrix[0].size() == 4);
    
    // Check symmetry
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(matrix[i][j] == matrix[j][i]);
        }
    }
    
    std::cout << "KroneckerFabric tests passed!" << std::endl;
}

void test_engine() {
    std::cout << "Testing VedicEngine..." << std::endl;
    
    vedic::VedicEngine engine(2);
    
    assert(engine.get_dimension() == 2);
    assert(engine.get_iteration() == 0);
    assert(engine.get_time() == 0.0);
    
    auto quantum_state = engine.get_quantum_state();
    assert(quantum_state.size() == 4); // 2^2 = 4
    
    // Check normalization
    double total_prob = 0.0;
    for (const auto& amp : quantum_state) {
        total_prob += std::norm(amp);
    }
    assert(std::abs(total_prob - 1.0) < 1e-12);
    
    std::cout << "VedicEngine tests passed!" << std::endl;
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "VEDIC ENGINE COMPREHENSIVE TEST SUITE" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        test_fixed256();
        test_rational();
        test_sutras();
        test_fabric();
        test_engine();
        
        std::cout << "\n=========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}