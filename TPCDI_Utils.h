#ifndef ARRAYFIRE_TPCDI_TPCDI_UTILS_H
#define ARRAYFIRE_TPCDI_TPCDI_UTILS_H
#include <arrayfire.h>
#include <iostream>
#include <tuple>
#include "Enums.h"

template<typename T>
inline void print(T i) { af::sync(); std::cout << i << std::endl; }
void printStr(af::array str_array);

namespace TPCDI_Utils {
    std::string loadFile(char const *filename);
    std::string collect(std::vector<std::string> const &files);
    af::array stringToNum(af::array &numstr, af::dtype type);
    af::array flipdims(af::array const &arr);
    constexpr std::pair<int8_t,int8_t> dateDelimIndices(DateFormat format) {
        if (format == YYYYDDMM || format == YYYYMMDD) return { 4, 7 };
        return { 2, 5 };
    }
    inline af::array endDate() {
        return join(0, af::constant(9999, 1, u16), af::constant(12, 1, u16), af::constant(31, 1, u16));
    }
    af::array stringToDate(af::array const &datestr, bool isDelimited = false, DateFormat dateFormat = YYYYMMDD);
    af::array dehashDate(af::array const &dateHash, DateFormat format);
    af::array stringToTime(af::array const &timestr, bool isDelimited = false);
    af::array stringToDateTime(af::array &datetimestr, bool isDelimited = false, DateFormat dateFormat = YYYYMMDD);
    af::array stringToBoolean(af::array &boolstr);
    af::array dateHash(af::array const &date);
    inline af::array timeHash(af::array const &time) { return dateHash(time); };
    af::array datetimeHash(af::array const &datetime);
    af::array prefixHash(af::array const &column);
    af::array polyHash(af::array const &column);

}

#endif //ARRAYFIRE_TPCDI_TPCDI_UTILS_H
