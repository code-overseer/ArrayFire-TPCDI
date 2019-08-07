//
// Created by Bryan Wong on 2019-08-03.
//

#ifndef ARRAYFIRE_TPCDI_AFTYPES_H
#define ARRAYFIRE_TPCDI_AFTYPES_H
#include <arrayfire.h>
#include <string>
#include "include/Enums.h"
template<af::dtype> struct GetType;
template<> struct GetType<b8> { static constexpr bool value = false; };
template<> struct GetType<u8> { static constexpr unsigned char value = 0; };
template<> struct GetType<f32> { static constexpr float value = 0; };
template<> struct GetType<f64> { static constexpr double value = 0;};
template<> struct GetType<u16> { static constexpr unsigned short value = 0; };
template<> struct GetType<s16> { static constexpr short value = 0; };
template<> struct GetType<u32> { static constexpr unsigned int value = 0; };
template<> struct GetType<s32> { static constexpr int value = 0; };
template<> struct GetType<u64> { static constexpr unsigned long long value = 0; };
template<> struct GetType<s64> { static constexpr long long value = 0; };
struct Time;
struct Date;
struct DateTime;
template<typename T> struct GetAFType;
template<> struct GetAFType<bool> {
    static constexpr af::dtype af_type = b8;
    static constexpr DataType df_type = DataType::BOOL;
    static constexpr char const* str = "bool";
};
template<> struct GetAFType<unsigned char> {
    static constexpr af::dtype af_type = u8;
    static constexpr DataType df_type = DataType::UCHAR;
    static constexpr char const *str = "uchar";
};
template<> struct GetAFType<float> {
    static constexpr af::dtype af_type = f32;
    static constexpr DataType df_type = DataType::FLOAT;
    static constexpr char const *str = "float";
};
template<> struct GetAFType<double> {
    static constexpr af::dtype af_type = f64;
    static constexpr DataType df_type = DataType::DOUBLE;
    static constexpr char const *str = "double";
};
template<> struct GetAFType<unsigned short> {
    static constexpr af::dtype af_type = u16;
    static constexpr DataType df_type = DataType::USHORT;
    static constexpr char const *str = "ushort";
};
template<> struct GetAFType<short> {
    static constexpr af::dtype af_type = s16;
    static constexpr DataType df_type = DataType::SHORT;
    static constexpr char const *str = "short";
};
template<> struct GetAFType<unsigned int> {
    static constexpr af::dtype af_type = u32;
    static constexpr DataType df_type = DataType::UINT;
    static constexpr char const *str = "uint";
};
template<> struct GetAFType<int> {
    static constexpr af::dtype af_type = s32;
    static constexpr DataType df_type = DataType::INT;
    static constexpr char const *str = "int";
};
template<> struct GetAFType<unsigned long long> {
    static constexpr af::dtype af_type = u64;
    static constexpr DataType df_type = DataType::ULONG;
    static constexpr char const *str = "ulong";
};
template<> struct GetAFType<long long> {
    static constexpr af::dtype af_type = s64;
    static constexpr DataType df_type = DataType::LONG;
    static constexpr char const *str = "long";
};
template<> struct GetAFType<Time> {
    static constexpr af::dtype af_type = u16;
    static constexpr DataType df_type = DataType::TIME;
    static constexpr char const *str = "time";
};
template<> struct GetAFType<Date> {
    static constexpr af::dtype af_type = u16;
    static constexpr DataType df_type = DataType::DATE;
    static constexpr char const *str = "date";
};
template<> struct GetAFType<DateTime> {
    static constexpr af::dtype af_type = u16;
    static constexpr DataType df_type = DataType::DATETIME;
    static constexpr char const *str = "datetime";
};

#endif //ARRAYFIRE_TPCDI_AFTYPES_H
