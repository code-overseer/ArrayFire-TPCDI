#include "include/AFDataFrame.h"
#include "include/BatchFunctions.h"
#include "include/TPCDI_Utils.h"
#include "include/Enums.h"
#include <cstring>
#include "include/Logger.h"
#if defined(USING_OPENCL)
#include "include/OpenCL/opencl_kernels.h"
#elif defined(USING_CUDA)
#include "CUDA/cuda_kernels.h"
#else
#include "CPU/vector_functions.h"
#endif
#ifndef ULL
#define ULL
    typedef unsigned long long ull;
#endif
using namespace BatchFunctions;
using namespace TPCDI_Utils;
using namespace af;


AFDataFrame::AFDataFrame(AFDataFrame&& other) noexcept : _deviceData(std::move(other._deviceData)),
                                                         _dataTypes(std::move(other._dataTypes)),
                                                         _nameToIdx(std::move(other._nameToIdx)),
                                                         _idxToName(std::move(other._idxToName)),
                                                         _name(std::move(other._name)) {
}

AFDataFrame::AFDataFrame(AFDataFrame const &other) : _deviceData(other._deviceData),
                                                     _dataTypes(other._dataTypes),
                                                     _nameToIdx(other._nameToIdx),
                                                     _idxToName(other._idxToName),
                                                     _name(other._name) {
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame&& other) noexcept {
    _deviceData = std::move(other._deviceData);
    _dataTypes = std::move(other._dataTypes);
    _nameToIdx = std::move(other._nameToIdx);
    _idxToName = std::move(other._idxToName);
    _name = std::move(other._name);
    return *this;
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame const &other) noexcept {
    _deviceData = other._deviceData;
    _dataTypes = other._dataTypes;
    _nameToIdx = other._nameToIdx;
    _name = other._name;
    return *this;
}

void AFDataFrame::add(array &column, DataType type, const char *name) {
    add(std::move(column), type, name);
}

void AFDataFrame::add(array &&column, DataType type, const char *name) {
    _deviceData.emplace_back(column);
    _dataTypes.emplace_back(type);

    if (name) nameColumn(name, _deviceData.size() - 1);
}

void AFDataFrame::insert(array &column, DataType type, int index, const char *name) {
    insert(std::move(column), type, index, name);
}

void AFDataFrame::insert(array &&column, DataType type, int index, const char *name) {
    _idxToName.erase(index);
    for (auto &i : _nameToIdx) {
        if (i.second >= index) {
            i.second += 1;
            _idxToName.erase(i.second);
            _idxToName.insert(std::make_pair(i.second, i.first));
        }
    }
    _deviceData.insert(_deviceData.begin() + index, column);
    _dataTypes.insert(_dataTypes.begin() + index, type);
    if (name) nameColumn(name, index);
}

void AFDataFrame::remove(int index) {
    _deviceData.erase(_deviceData.begin() + index);
    _dataTypes.erase(_dataTypes.begin() + index);
    _nameToIdx.erase(_idxToName[index]);
    _idxToName.erase(index);
    for (auto &i : _nameToIdx) {
        if (i.second >= index) {
            i.second -= 1;
            _idxToName.erase(i.second);
            _idxToName.insert(std::make_pair(i.second, i.first));
        }
    }
}

void AFDataFrame::_flush(af::array const &idx) {
    for (auto &i : _deviceData) {
        i = i(span, idx);
        i.eval();
    }
}

af::array AFDataFrame::stringMatch(int column, char const *str) const {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Invalid column type");
    auto length = strlen(str) + 1;
    if (_deviceData[column].dims(0) < length) return constant(0, dim4(1, _deviceData[column].dims(1)), b8);

    auto match = array(length, str);
    auto idx = allTrue(batchFunc(_deviceData[column].rows(0, --length), match, batchEqual), 0);
    idx.eval();
    return idx;
}

AFDataFrame AFDataFrame::project(int const *columns, int size, std::string const &name) const {
    AFDataFrame output;
    output.name(name);
    for (int i = 0; i < size; i++) {
        int n = columns[i];
        output.add(project(n), _dataTypes[n]);
        if (_idxToName.count(n)) output.nameColumn(_idxToName.at(n), i);
    }
    return output;
}

AFDataFrame AFDataFrame::select(af::array const &index) const {
    AFDataFrame out(*this);
    out._flush(index);
    return out;
}

AFDataFrame AFDataFrame::project(std::string const *names, int size, std::string const &name) const {
    int columns[size];
    for (int i = 0; i < size; i++) {
        columns[i] = _nameToIdx.at(names[i]);
    }
    return project(columns, size, name);
}

AFDataFrame AFDataFrame::zip(AFDataFrame &&rhs) const {
    if (length() != rhs.length()) throw std::runtime_error("Left and Right tables do not have the same length");
    AFDataFrame output = *this;

    for (size_t i = 0; i < rhs._deviceData.size(); ++i) {
        output.add(rhs._deviceData[i], rhs._dataTypes[i], (rhs.name() + "." + rhs._idxToName.at(i)).c_str());
    }
    return output;
}

AFDataFrame AFDataFrame::concatenate(AFDataFrame &&frame) const {
    if (_dataTypes.size() != frame._dataTypes.size()) throw std::runtime_error("Number of attributes do not match");
    for (size_t i = 0; i < _dataTypes.size(); i++) {
        if (frame._dataTypes[i] != _dataTypes[i])
            throw std::runtime_error("Attribute types do not match");
    }
    auto out = *this;
    for (size_t i = 0; i < out._deviceData.size(); ++i) {
        if (out._dataTypes[i] == STRING) {
            auto delta = out._deviceData[i].dims(0) - frame._deviceData[i].dims(0);
            if (delta > 0) {
                auto back = constant(0, delta, frame._deviceData[i].dims(1), u8);
                frame._deviceData[i] = join(0, frame._deviceData[i], back);
            } else if (delta < 0) {
                delta = -delta;
                auto back = constant(0, delta, out._deviceData[i].dims(1), u8);
                out._deviceData[i] = join(0, out._deviceData[i], back);
            }
        }
        out._deviceData[i] = join(1, out._deviceData[i], frame._deviceData[i]);
    }
    return out;
}

void AFDataFrame::sortBy(int column, bool isAscending) {
    array elements = hashColumn(column, true);
    auto const size = elements.dims(0);
    if (!size) return;
    array sorting;
    array idx;
    sort(sorting, idx, elements(end, span), 1, isAscending);
    for (auto j = size - 2; j >= 0; --j) {
        elements = elements(span, idx);
        sort(sorting, idx, elements(j, span), 1, isAscending);
    }
    _flush(idx);
}

void AFDataFrame::sortBy(int *columns, int size, bool const *isAscending) {
    for (auto i = size - 1; i >= 0; --i) {
        auto asc = isAscending ? isAscending[i] : true;
        sortBy(columns[i], asc);
    }
}

array AFDataFrame::hashColumn(af::array const &column, DataType type, bool sortable) {
    if (type == STRING) return sortable ? byteHash(column) : polyHash(byteHash(column));
    if (type == DATE) return dateHash(column).as(u64);
    if (type == TIME) return timeHash(column).as(u64);
    if (type == DATETIME) return datetimeHash(column).as(u64);

    return array(column).as(u64);
}

AFDataFrame AFDataFrame::equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const {
    auto &leftType = _dataTypes[lhs_column];
    auto &rightType = rhs._dataTypes[rhs_column];
    if (leftType != rightType) throw std::runtime_error("Supplied column data types do not match");
    auto l = hashColumn(lhs_column);
    auto r = rhs.hashColumn(rhs_column);

    auto idx = setCompare(l, r);
    if (_deviceData[lhs_column].dims(0) <= rhs._deviceData[rhs_column].dims(0)) {
        l = _deviceData[lhs_column](span, idx.first);
        r = rhs._deviceData[rhs_column](range(dim4(l.dims(0)),0,u32), idx.second);
        r = moddims(r, l.dims());
    } else {
        r = rhs._deviceData[rhs_column](span, idx.second);
        l = _deviceData[lhs_column](range(dim4(r.dims(0)),0,u32), idx.first);
        l = moddims(l, r.dims());
    }
    {
        /* Collision Check */
        auto tmp = where(allTrue(l == r, 0));
        idx.first = idx.first(tmp);
        idx.second = idx.second(tmp);
    }

    AFDataFrame result;
    for (size_t i = 0; i < _deviceData.size(); i++) {
      result.add(_deviceData[i](span, idx.first), _dataTypes[i], _idxToName.at(i).c_str());
    }

    for (size_t i = 0; i < rhs._deviceData.size(); i++) {
        if (i == rhs_column) continue;
        result.add(rhs._deviceData[i](span, idx.second), rhs._dataTypes[i],
		   (rhs.name() + "." + rhs._idxToName.at(i)).c_str());
    }

    return result;
}

std::pair<af::array, af::array> AFDataFrame::setCompare(array const &left, array const &right) {
    printf("LHS rows: %llu\n", left.elements());
    printf("RHS rows: %llu\n", right.elements());
    Logger::startTimer("Join");
    array lhs;
    array rhs;
    array idx;
    sort(lhs, idx, left, 1);
    lhs = join(0, lhs, idx.as(lhs.type()));
    sort(rhs, idx, right, 1);
    rhs = join(0, rhs, idx.as(rhs.type()));

    auto const equalSet = flipdims(setIntersect(setUnique(lhs.row(0), true), setUnique(rhs.row(0), true), true));
    bagSetIntersect(lhs, equalSet);
    bagSetIntersect(rhs, equalSet);

    auto equals = equalSet.elements();
    joinScatter(lhs, rhs, equals);
    printf("Output rows: %llu\n", lhs.elements());
    Logger::logTime("Join");
    return { lhs, rhs };
}

std::string AFDataFrame::name(std::string const& str) {
    _name = str;
    return _name;
}

void AFDataFrame::nameColumn(std::string const& name, int column) {
    if (_idxToName.count(column)) _nameToIdx.erase(_idxToName.at(column));
    _nameToIdx[name] = column;
    _idxToName[column] = name;
}

void AFDataFrame::flushToHost() {
    af::sync();
    if (_deviceData.empty()) return;
    for (auto const &a : _deviceData) {
        if (a.bytes()) {
            auto tmp = malloc(a.bytes());
            a.host(tmp);
            _hostData.emplace_back(tmp);
        } else {
            _hostData.emplace_back(nullptr);
        }
    }
    _deviceData.clear();
}

AFDataFrame::~AFDataFrame() {
    if (_hostData.empty()) return;
    auto i = (uint64_t*) _hostData[0];
    for (auto &dat : _hostData) freeHost(dat);
    _dataTypes.clear();
}

void AFDataFrame::clear() {
    _deviceData.clear();
    _name.clear();
    _idxToName.clear();
    _nameToIdx.clear();
    if (!_hostData.empty()) {
        for (auto &dat : _hostData) freeHost(dat);
    }
    _hostData.clear();
    _dataTypes.clear();
}



