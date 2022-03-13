#pragma once

#include <Corrade/TestSuite/Tester.h>

#include <math/Vector.hpp>

using namespace Corrade;

template<count_t dim, typename T>
struct TestVector : TestSuite::Tester
{
	using Vec = Vector<dim, T>;

	explicit TestVector() {
		addTests({&TestVector<dim, T>::zeroConstruction});
		addTests({&TestVector<dim, T>::valueConstruction});
		addTests({&TestVector<dim, T>::piecewiseOperators});
	}

	void zeroConstruction() {
		Vec defaultInitialized;
		Vec emptyInitialized {};
		Vec zeroInitialized {static_cast<T>(0)};
		for (count_t i = 0; i < dim; ++i) {
			CORRADE_COMPARE(defaultInitialized[i], 0);
			CORRADE_COMPARE(emptyInitialized[i], 0);
			CORRADE_COMPARE(zeroInitialized[i], 0);
		}
	}

	void valueConstruction() {
		Vec oneInitialized {static_cast<T>(1)};
		for (count_t i = 0; i < dim; ++i) {
			CORRADE_COMPARE(oneInitialized[i], 1);
		}
		if constexpr (dim == 2) {
			Vec oneTwo = {1 ,2};
			CORRADE_COMPARE(oneTwo[0], 1);
			CORRADE_COMPARE(oneTwo[1], 2);
		}
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
			CORRADE_COMPARE(negSixes[i], -6);
		}
	}
};
