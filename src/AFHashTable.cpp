#include "AFHashTable.h"
#include "Column.h"
#include "Utils.h"
#include <cmath>
#include <exception>

bool AFHashTable::_isPrime(const ull &x) {
    uint32_t div = 3;
    while (div * div <= x) {
        if (!(x % div)) return false;
        div += 2;
    }
    return true;
}

unsigned int AFHashTable::_getPrime(ull x) {
    x = x | 1Ull;
    while (!_isPrime(x)) x+= 2Ull;
    if (x > UINT32_MAX) throw std::runtime_error("HashTable size limit exceeded");
    return x;
}

AFHashTable::AFHashTable(Column const &col) {
    _values = Utils::hflat(af::setUnique(col.data().as(u64), false));
    _generate();
}

void AFHashTable::_generate() {
    _values.eval();
    _buckets = _getPrime(_values.elements());

    _occ = constant(0, af::dim4(1, _buckets), _occ.type());
    auto keys = (_values % _buckets).as(u64);
    af::sort(keys, _values, keys, _values, 1);

    auto bins = af::accum(af::join(1, af::constant(0, 1, b8), diff1(keys, 1) > 0), 1);
    bins = Utils::hflat(histogram(bins, bins.elements())).as(_occ.type());
    bins = bins(bins > 0);
    _occ(setUnique(keys, true)) = bins;
    _occ.eval();

    _ptr = af::scan(_occ.as(_ptr.type()), 1, AF_BINARY_ADD, false);
    _ptr.eval();
}

AFHashTable::AFHashTable(af::array &&set) {
    if (set.type() != u64) throw std::runtime_error("Expected unsigned int64 array");
    _values = std::move(set);
    _values = Utils::hflat(_values);
    _generate();
}

AFHashTable::AFHashTable(af::array const &set) {
    if (set.type() != u64) throw std::runtime_error("Expected unsigned int64 array");
    _values = set;
    _values = Utils::hflat(_values);
    _generate();
}



