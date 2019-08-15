#include "include/AFHashTable.h"
#include "include/Column.h"
#include "include/TPCDI_Utils.h"
#include <cmath>
#include <exception>

bool AFHashTable::_isPrime(const ull &x) {
    double ceil = std::sqrt(x);
    uint32_t div = 3;
    while (div <= ceil) {
        if (!(x % div)) return false;
        div += 2;
    }
    return true;
}

ull AFHashTable::_getPrime(ull x) {
    x = x | 1Ull;
    while (!_isPrime(x)) x+= 2Ull;
    return x;
}

AFHashTable::AFHashTable(Column const &col) {
    _values = TPCDI_Utils::hflat(af::setUnique(col.data().as(u64), false));
    _generate();
}


void AFHashTable::_generate() {
    _values.eval();
    _buckets = _getPrime(_values.elements());

    _occ = constant(0, af::dim4(1, _buckets), u64);
    auto keys = (_values % _buckets).as(u64);
    af::sort(keys, _values, keys, _values, 1);

    auto bins = af::accum(af::join(1, af::constant(0, 1, keys.type()), diff1(keys, 1) > 0), 1);
    bins = TPCDI_Utils::hflat(histogram(bins, bins.elements())).as(u32);
    bins = bins(bins > 0);
    _occ(setUnique(keys, true)) = bins ;
    _occ.eval();

    _ptr = af::scan(_occ, 1, AF_BINARY_ADD, false);
    _ptr.eval();
}

AFHashTable::AFHashTable(af::array &&set) {
    if (set.type() != u64) throw std::runtime_error("Expected unsigned int64 array");
    _values = std::move(set);
    _values = TPCDI_Utils::hflat(_values);
    _generate();
}



