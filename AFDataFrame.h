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
#include "Enums.h"
#include <functional>
#include "AFParser.hpp"
#include "BatchFunctions.h"

class AFDataFrame {
public:
    AFDataFrame() : _length(0) {};
    AFDataFrame(AFDataFrame&& other) noexcept;
    AFDataFrame(AFDataFrame const &other);
    virtual ~AFDataFrame() = default;
    void add(af::array &column, DataType type);
    void add(af::array &&column, DataType type);
    void insert(af::array &column, DataType type, int index);
    void insert(af::array &&column, DataType type, int index);
    void remove(int index);
    void remove(std::string const &name) { remove(_columnNames[name]); }
    static af::array dateOrTimeHash(const af::array &date);
    af::array dateOrTimeHash(int index) const ;
    af::array dateOrTimeHash(std::string const &name) const { return dateOrTimeHash(_columnNames.at(name)); }
    static af::array datetimeHash(const af::array &datetime);
    af::array datetimeHash(int index) const;
    af::array datetimeHash(std::string const &name) const { return datetimeHash(_columnNames.at(name)); }
    static af::array hashColumn(af::array const &column, DataType type, bool sortable = false);
    af::array hashColumn(int column, bool sortable = false) const;
    af::array hashColumn(std::string const &name, bool sortable = false) const { return hashColumn(_columnNames.at(name), sortable); }
    void dateSort(int index, bool ascending = true);
    void dateSort(std::string const &name, bool ascending = true) { return dateSort(_columnNames[name], ascending); }
    void stringLengthMatchSelect(int column, size_t length);
    void stringLengthMatch(std::string const &name, size_t len) { return stringLengthMatchSelect(_columnNames[name], len); }
    void stringMatchSelect(int column, char const *str);
    void stringMatch(std::string const &name, char const* str) { return stringMatchSelect(_columnNames[name], str); }
    void select(af::array const &index) { _rowIndexes = index; _flush();};
    AFDataFrame project(int const *columns, int size, std::string const &name) const;
    AFDataFrame project(std::string const *names, int size, std::string const &name) const;
    void concatenate(AFDataFrame &frame);
    void concatenate(AFDataFrame &&frame);
    static void printStr(af::array str);
    static af::array endDate();
    static af::array stringToDate(af::array &datestr, DateFormat inputFormat, bool isDelimited);
    static std::pair<af::array, af::array> innerJoin(af::array const &lhs, af::array const &rhs, af::batchFunc_t predicate = BatchFunctions::batchEqual);
    bool isEmpty();
    std::vector<af::array> &data() { return _deviceData; }
    af::array &data(int column) { return _deviceData[column]; }
    af::array &data(std::string const &name) { return data(_columnNames.at(name)); }
    std::vector<DataType> &types() { return _dataTypes; }
    DataType &types(int column) { return _dataTypes[column]; }
    DataType &types(std::string const &name) { return types(_columnNames.at(name)); }
    AFDataFrame& operator=(AFDataFrame&& other) noexcept;
    AFDataFrame& operator=(AFDataFrame const &other) noexcept;
    void reorder(int const *seq, int size);
    void reorder(std::string const *seq, int size);
    void sortBy(int column, bool isAscending = true);
    void sortBy(std::string const &name, bool isAscending = true) { sortBy(_columnNames.at(name), isAscending); }
    void sortBy(int *columns, int size, const bool *isAscending = nullptr);
    void sortBy(std::string *columns, int size, bool *isAscending = nullptr) {
        int seqnum[size];
        for (int j = 0; j < size; ++j) seqnum[j] = _columnNames[columns[j]];
        sortBy(seqnum, size, isAscending);
    }
    static af::array prefixHash(af::array const &column);
    af::array prefixHash(int column) const;
    af::array prefixHash(std::string const &name) const { return prefixHash(_columnNames.at(name)); }
    static af::array polyHash(af::array const &column);
    af::array polyHash(int column) const;
    af::array polyHash(std::string const &name) const { return polyHash(_columnNames.at(name)); }
    AFDataFrame equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const;
    AFDataFrame equiJoin(AFDataFrame const &rhs, std::string const &lName, std::string const &rName) const {
        return equiJoin(rhs, _columnNames.at(lName), rhs._columnNames.at(rName));
    }
    std::string name(const std::string& str);
    std::string name() const;
    void nameColumn(const std::string& name, int column);
    void nameColumn(const std::string& name, const std::string &old);
private:
    af::array project(int column) const;
    std::vector<af::array> _deviceData;
    std::vector<DataType> _dataTypes;
    std::string _tableName;
    unsigned long _length;
    af::array _rowIndexes;
    std::unordered_map<std::string, unsigned int> _columnNames;
    std::unordered_map<unsigned int, std::string> _columnToName;
    void _flush();
};
#endif //ARRAYFIRE_TPCDI_AFDATAFRAME_H
