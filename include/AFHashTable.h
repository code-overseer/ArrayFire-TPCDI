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
    af::array _occ = af::array(0, u64);
    unsigned int _buckets = 0;
    static bool _isPrime(ull const &x);
    static unsigned int _getPrime(ull x);
    void _generate();
public:
    inline void unlock() const { _values.unlock(); _ptr.unlock(); _occ.unlock(); }
    inline ull* values() const { return _values.device<ull>(); }
    inline ull* pointers() const { return _ptr.device<ull>(); }
    inline ull* occupancy() const { return _occ.device<ull>(); }
    inline ull buckets() const { return _buckets; }
    inline ull elements() const { return _values.elements(); }
    inline af::array const& getValues() const { return _values; }
    inline af::array const& getPtr() const { return _ptr; }
    inline af::array const& getOcc() const { return _occ; }
    inline af::array::array_proxy getValues(af::index const& i) const { return _values(i); }
    inline af::array::array_proxy getPtr(af::index const& i) const { return _ptr(i); }
    inline af::array::array_proxy getOcc(af::index const& i) const { return _occ(i); }
    explicit AFHashTable(Column const &col);
    explicit AFHashTable(af::array &&set);
    explicit AFHashTable(af::array const &set);
    ~AFHashTable() { unlock(); }

};

#endif //ARRAYFIRE_TPCDI_AFHASHTABLE_H
