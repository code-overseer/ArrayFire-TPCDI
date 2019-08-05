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
#include <functional>
#include "Enums.h"
#include "AFParser.hpp"
#include "BatchFunctions.h"
#include "Column.h"

class AFDataFrame {
public:
    AFDataFrame() = default;;
    AFDataFrame(AFDataFrame&& other) noexcept;
    AFDataFrame(AFDataFrame const &other);
    AFDataFrame& operator=(AFDataFrame&& other) noexcept;
    AFDataFrame& operator=(AFDataFrame const &other) noexcept;
    virtual ~AFDataFrame() = default;
    void add(Column &column, const char *name = nullptr);
    void add(Column &&column, const char *name = nullptr);
    void insert(Column &column, int index, const char *name = nullptr);
    void insert(Column &&column, int index, const char *name = nullptr);
    void remove(int index);
    af::array stringMatch(int column, char const *str) const;
    AFDataFrame select(af::array const &index, std::string const &name) const;
    AFDataFrame project(int const *columns, int size, std::string const &name) const;
    AFDataFrame project(std::string const *names, int size, std::string const &name) const;
    AFDataFrame unionize(AFDataFrame &frame) const;
    AFDataFrame unionize(AFDataFrame &&frame) const { return unionize(frame); }
    AFDataFrame zip(AFDataFrame &rhs) const;

    static std::pair<af::array, af::array> setCompare(af::array const &lhs, af::array const &rhs);
    void sortBy(int column, bool isAscending = true);
    void sortBy(int *columns, int size, const bool *isAscending = nullptr);
    AFDataFrame equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const;
    void nameColumn(const std::string& name, unsigned int column);
    std::string name(const std::string& str);
    void flushToHost();
    void clear();

    inline void remove(std::string const &name) { remove(_nameToIdx[name]); }
    inline af::array hashColumn(int const column, bool sortable = false) const { return _data[column].hash(sortable); }
    inline af::array hashColumn(std::string const &name, bool sortable = false) const { return hashColumn(_nameToIdx.at(name), sortable); }
    inline af::array stringMatch(std::string const &name, char const *str) const { return stringMatch(_nameToIdx.at(name), str); }
    inline AFDataFrame zip(AFDataFrame &&rhs) const { return zip(rhs); }
    inline AFDataFrame concatenate(AFDataFrame &frame) const { return unionize(std::move(frame)); }
    inline bool isEmpty() { return _data.empty() || _data[0].isempty(); }
    inline std::vector<Column> &data() { return _data; }
    inline Column &data(unsigned int column) { return _data[column]; }
    inline Column &data(std::string const &name) { return data(_nameToIdx.at(name)); }
    inline AFDataFrame equiJoin(AFDataFrame const &rhs, std::string const &lName, std::string const &rName) const { return equiJoin(rhs, _nameToIdx.at(lName), rhs._nameToIdx.at(rName)); }
    inline void sortBy(std::string const &name, bool isAscending = true) { sortBy(_nameToIdx.at(name), isAscending); }
    inline void sortBy(std::string const *columns, int const size, bool const *isAscending = nullptr) {
        int seqnum[size];
        for (int j = 0; j < size; ++j) seqnum[j] = _nameToIdx[columns[j]];
        sortBy(seqnum, size, isAscending);
    }
    inline uint64_t length() const { return _data.empty() ? 0 : _data[0].dims(1); }
    inline void nameColumn(const std::string& name, const std::string &old) { nameColumn(name, _nameToIdx.at(old)); }
    inline std::string name() const { return _name; }
private:
    inline Column project(int column) const { return Column(_data[column]); }
    std::vector<Column> _data;
    std::string _name;
    std::unordered_map<std::string, unsigned int> _nameToIdx;
    std::unordered_map<unsigned int, std::string> _idxToName;
    void _flush(af::array const &idx);
};
#endif //ARRAYFIRE_TPCDI_AFDATAFRAME_H
