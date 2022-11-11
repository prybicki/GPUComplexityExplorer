#pragma once

#include <vector>

#include <Corrade/TestSuite/Tester.h>

#include <math/Vector.hpp>

using namespace Corrade;

// TODO:
// - compile all variants to a single file
// - make it easy to test cases
template<count_t dim, typename T>
struct TestVector : TestSuite::Tester
{
	using Vec = Vector<dim, T>;
	std::array<T, dim> values;

	explicit TestVector() {
		std::iota(values.begin(), values.end(), 1);
		addTests({&TestVector<dim, T>::zeroConstruction});
		addTests({&TestVector<dim, T>::valueConstruction});
		addTests({&TestVector<dim, T>::piecewiseOperators});
		addTests({&TestVector<dim, T>::iteration});
	}

	void zeroConstruction() {
		Vec defaultInitialized;
		Vec emptyInitialized {};
		Vec zeroInitialized {static_cast<T>(0)};
		for (count_t i = 0; i < dim; ++i) {
			CORRADE_COMPARE(defaultInitialized[i], static_cast<T>(0));
			CORRADE_COMPARE(emptyInitialized[i],   static_cast<T>(0));
			CORRADE_COMPARE(zeroInitialized[i],    static_cast<T>(0));
		}
	}

	void valueConstruction() {
		Vec oneInitialized {static_cast<T>(1)};
		for (count_t i = 0; i < dim; ++i) {
			CORRADE_COMPARE(oneInitialized[i], 1);
		}
	}

	void iteration() {
		int index = 0;
		Vec vector = values;
		for (auto&& value : vector) {
			CORRADE_COMPARE(value, values[index]);
			index += 1;
		}
		CORRADE_COMPARE(index, dim);
	}

	void piecewiseOperators() {
		Vec ones = {static_cast<T>(1)};
		Vec twos = {static_cast<T>(2)};

		Vec threes = ones + twos;
		Vec negOnes = ones - twos;
		Vec sixes = threes * twos;
		Vec negSixes = sixes / negOnes;
		for (count_t i = 0; i < dim; ++i) {
			CORRADE_COMPARE(threes[i],     3);
			CORRADE_COMPARE(negOnes[i],   -1);
			CORRADE_COMPARE(sixes[i],      6);
			CORRADE_COMPARE(negSixes[i],  -6);
		}
	}
};
