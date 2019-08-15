#ifndef ARRAYFIRE_TPCDI_AFHASHTABLE_H
#define ARRAYFIRE_TPCDI_AFHASHTABLE_H
#include <arrayfire.h>
#ifndef ULL
#define ULL
typedef unsigned long long ull;
#endif
class Column;

class AFHashTable {
private:
    af::array _values = af::array(0, u64);
    af::array _ptr = af::array(0, u64);
    af::array _occ = af::array(0, u32);
    ull _buckets = 0;
    static bool _isPrime(ull const &x);
    static ull _getPrime(ull x);
public:
    inline void unlock() {_values.unlock(); _ptr.unlock(); _occ.unlock(); }
    inline ull* data() { return _values.device<ull>(); }
    inline ull* pointers() { return _ptr.device<ull>(); }
    inline unsigned int* occupancy() { return _occ.device<unsigned int>(); }
    inline ull buckets() { return _buckets; }
    explicit AFHashTable(Column const &col);
    explicit AFHashTable(af::array &&set);
    ~AFHashTable() { unlock(); }

    void _generate();
};

#endif //ARRAYFIRE_TPCDI_AFHASHTABLE_H
