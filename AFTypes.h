//
// Created by Bryan Wong on 2019-08-03.
//

#ifndef ARRAYFIRE_TPCDI_AFTYPES_H
#define ARRAYFIRE_TPCDI_AFTYPES_H
#include <arrayfire.h>
#include <string>

template<af::dtype> struct GetType;
template<> struct GetType<b8> { static constexpr bool value = 0; };
template<> struct GetType<u8> { static constexpr unsigned char value = 0; };
template<> struct GetType<f32> { static constexpr float value = 0; };
template<> struct GetType<f64> { static constexpr double value = 0;};
template<> struct GetType<u32> { static constexpr unsigned int value = 0; };
template<> struct GetType<s32> { static constexpr int value = 0; };
template<> struct GetType<u64> { static constexpr unsigned long long value = 0; };
template<> struct GetType<s64> { static constexpr long long value = 0; };

template<typename T> struct GetAFType;
template<> struct GetAFType<bool> {
    static constexpr af::dtype value = b8;
    static constexpr char const* str = "bool";
};
template<> struct GetAFType<unsigned char> {
    static constexpr af::dtype value = u8;
    static constexpr char const *str = "uchar";
};
template<> struct GetAFType<float> {
    static constexpr af::dtype value = f32;
    static constexpr char const *str = "float";
};
template<> struct GetAFType<double> {
    static constexpr af::dtype value = f64;
    static constexpr char const *str = "double";
};
template<> struct GetAFType<unsigned int> {
    static constexpr af::dtype value = u32;
    static constexpr char const *str = "uint";
};
template<> struct GetAFType<int> {
    static constexpr af::dtype value = s32;
    static constexpr char const *str = "int";
};
template<> struct GetAFType<unsigned long long> {
    static constexpr af::dtype value = u64;
    static constexpr char const *str = "ulong";
};
template<> struct GetAFType<long long> {
    static constexpr af::dtype value = s64;
    static constexpr char const *str = "long";
};

#endif //ARRAYFIRE_TPCDI_AFTYPES_H
