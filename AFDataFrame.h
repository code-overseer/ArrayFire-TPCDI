//
// Created by Bryan Wong on 2019-06-27.
//

#ifndef ARRAYFIRE_TPCDI_AFDATAFRAME_H
#define ARRAYFIRE_TPCDI_AFDATAFRAME_H
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <arrayfire.h>

class AFDataFrame {
public:
    enum DataType { INT, SHORT, LONG, UINT, UCHAR, USHORT, U64, FLOAT, DOUBLE, STRING, BOOL, DATE, TIME };
    void add(af::array &column, DataType type);
    void insert(af::array &column, DataType type, int index);
    void remove(int index);
    af::array dateKeyGen(int index);
    void dateSort(int index, bool ascending = true);
    void stringLengthMatch(int column, size_t len);
    void stringMatch(int column, char const* str);
    void concatenate(AFDataFrame &frame);
    static void printStr(af::array str);
    static af::array endDate();
    virtual ~AFDataFrame() = default;
    bool isEmpty();
    std::vector<af::array> &data();
    AFDataFrame() : _length(0) {};
    AFDataFrame(AFDataFrame&& other) noexcept;
private:
    std::vector<af::array> _deviceData;
    std::vector<DataType> _dataTypes;
    unsigned long _length;
    af::array _rowIndexes;
    std::unordered_map<std::string, unsigned int> _columnNames;
    void _flush();
};


#endif //ARRAYFIRE_TPCDI_AFDATAFRAME_H
