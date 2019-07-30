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
#if USING_OPENCL
#include "OpenCL/opencl_kernels.h"
#elif USING_CUDA
#include "CUDA/cuda_kernels.h"
#endif

class AFDataFrame {
public:
    AFDataFrame() = default;;
    AFDataFrame(AFDataFrame&& other) noexcept;
    AFDataFrame(AFDataFrame const &other);
    AFDataFrame& operator=(AFDataFrame&& other) noexcept;
    AFDataFrame& operator=(AFDataFrame const &other) noexcept;
    virtual ~AFDataFrame();
    void add(af::array &column, DataType type, const char *name = nullptr);
    void add(af::array &&column, DataType type, const char *name = nullptr);
    void insert(af::array &column, DataType type, int index, const char *name = nullptr);
    void insert(af::array &&column, DataType type, int index, const char *name = nullptr);
    void remove(int index);
    void remove(std::string const &name) { remove(_nameToIdx[name]); }
    af::array stringMatchIdx(int column, char const *str) const;
    AFDataFrame select(af::array const &index) const;
    AFDataFrame project(int const *columns, int size, std::string const &name) const;
    AFDataFrame project(std::string const *names, int size, std::string const &name) const;
    void concatenate(AFDataFrame &&frame);
    AFDataFrame zip(AFDataFrame &&rhs) const;
    static af::array hashColumn(af::array const &column, DataType type, bool sortable = false);
    static std::pair<af::array, af::array> crossCompare(af::array const &lhs, af::array const &rhs,
                                                        af::batchFunc_t predicate = BatchFunctions::batchEqual);
    static std::pair<af::array, af::array> setCompare(af::array lhs, af::array rhs);
    void sortBy(int column, bool isAscending = true);
    void sortBy(int *columns, int size, const bool *isAscending = nullptr);
    AFDataFrame equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const;
    void nameColumn(const std::string& name, int column);
    std::string name(const std::string& str);
    void flushToHost();
    void clear();

    inline af::array hashColumn(int const column, bool sortable = false) const { return hashColumn(_deviceData[column], _dataTypes[column], sortable); }
    inline af::array hashColumn(std::string const &name, bool sortable = false) const { return hashColumn(_nameToIdx.at(name), sortable); }
    inline af::array stringMatchIdx(std::string const &name, char const *str) const{ return stringMatchIdx(_nameToIdx.at(name), str); }
    inline AFDataFrame zip(AFDataFrame &rhs) const { return zip(std::move(rhs)); }
    inline void concatenate(AFDataFrame &frame) { concatenate(std::move(frame)); }
    inline bool isEmpty() { return _deviceData.empty(); }
    inline std::vector<af::array> &data() { return _deviceData; }
    inline af::array &data(int column) { return _deviceData[column]; }
    inline af::array &data(std::string const &name) { return data(_nameToIdx.at(name)); }
    inline std::vector<DataType> &types() { return _dataTypes; }
    inline DataType &types(int column) { return _dataTypes[column]; }
    inline DataType &types(std::string const &name) { return types(_nameToIdx.at(name)); }
    inline AFDataFrame equiJoin(AFDataFrame const &rhs, std::string const &lName, std::string const &rName) const { return equiJoin(rhs, _nameToIdx.at(lName), rhs._nameToIdx.at(rName)); }
    inline void sortBy(std::string const &name, bool isAscending = true) { sortBy(_nameToIdx.at(name), isAscending); }
    inline void sortBy(std::string const *columns, int const size, bool const *isAscending = nullptr) {
        int seqnum[size];
        for (int j = 0; j < size; ++j) seqnum[j] = _nameToIdx[columns[j]];
        sortBy(seqnum, size, isAscending);
    }
    inline uint64_t length() const { return _deviceData.empty() ? 0 : _deviceData[0].dims(1); }
    inline void nameColumn(const std::string& name, const std::string &old) { nameColumn(name, _nameToIdx.at(old)); }
    inline std::string name() const { return _tableName; }
private:
    static af::array _subSort(af::array const &elements, af::array const &bucket, bool isAscending);
    inline af::array project(int column) const { return af::array(_deviceData[column]); }
    std::vector<af::array> _deviceData;
    std::vector<void*> _hostData;
    std::vector<DataType> _dataTypes;
    std::string _tableName;
    std::unordered_map<std::string, unsigned int> _nameToIdx;
    std::unordered_map<unsigned int, std::string> _idxToName;
    void _flush(af::array const &idx);
    static void _removeNonExistant(const af::array &setrl, af::array &lhs, af::array &rhs);
    static void _removeNonExistant(const af::array &setrl, af::array &lhs, af::array &rhs, bool swt);
};
#endif //ARRAYFIRE_TPCDI_AFDATAFRAME_H
