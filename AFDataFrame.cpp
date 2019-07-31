#include "AFDataFrame.h"
#include "BatchFunctions.h"
#include "TPCDI_Utils.h"
#include "Enums.h"
#include <cstring>
#include "Logger.h"
#if defined(USING_OPENCL)
#include "OpenCL/opencl_kernels.h"
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
                                                         _tableName(std::move(other._tableName)) {
}

AFDataFrame::AFDataFrame(AFDataFrame const &other) : _deviceData(other._deviceData),
                                                    _dataTypes(other._dataTypes),
                                                    _nameToIdx(other._nameToIdx),
                                                    _idxToName(other._idxToName),
                                                     _tableName(other._tableName) {
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame&& other) noexcept {
    _deviceData = std::move(other._deviceData);
    _dataTypes = std::move(other._dataTypes);
    _nameToIdx = std::move(other._nameToIdx);
    _idxToName = std::move(other._idxToName);
    _tableName = std::move(other._tableName);
    return *this;
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame const &other) noexcept {
    _deviceData = other._deviceData;
    _dataTypes = other._dataTypes;
    _nameToIdx = other._nameToIdx;
    _tableName = other._tableName;
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

    auto match = array(dim4(length, 1), str);
    auto idx = allTrue(batchFunc(_deviceData[column].rows(0, length), match, batchEqual), 0);
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

void AFDataFrame::concatenate(AFDataFrame &&frame) {
    if (_dataTypes.size() != frame._dataTypes.size()) throw std::runtime_error("Number of attributes do not match");
    for (int i = 0; i < _dataTypes.size(); i++) {
        if (frame._dataTypes[i] != _dataTypes[i])
            throw std::runtime_error("Attribute types do not match");
    }

    for (int i = 0; i < _deviceData.size(); ++i) {

        if (_dataTypes[i] == STRING) {
            auto delta = _deviceData[i].dims(0) - frame._deviceData[i].dims(0);
            if (delta > 0) {
                auto back = constant(0, delta, frame._deviceData[i].dims(1), u8);
                frame._deviceData[i] = join(0, frame._deviceData[i], back);
            } else if (delta < 0) {
                delta = -delta;
                auto back = constant(0, delta, _deviceData[i].dims(1), u8);
                _deviceData[i] = join(0, _deviceData[i], back);
            }
        }
        _deviceData[i] = join(1, _deviceData[i], frame._deviceData[i]);
    }
}

void AFDataFrame::sortBy(int column, bool isAscending) {
    array elements = hashColumn(column, true);
    auto const size = elements.dims(0);
    array sorting;
    array idx;
    sort(sorting, idx, elements(0, span), 1, isAscending);
    _flush(idx);
    for (int j = 1; j < size; ++j) {
        idx = _subSort(elements(j, span), elements(j - 1, span), isAscending);
        _flush(idx);
    }
}

array AFDataFrame::_subSort(array const &elements, array const &bucket, bool const isAscending) {
    auto idx_output = isAscending ? diff1(bucket, 1).as(s64) : diff1(flip(bucket, 1), 1).as(s64);
    idx_output = join(1, constant(1, dim4(1), u64), (idx_output > 0).as(u64));
    idx_output = accum(idx_output, 1) - 1;
    idx_output = flipdims(histogram(idx_output, idx_output.elements())).as(u64);
    idx_output = idx_output(idx_output > 0);
    idx_output = join(0, scan(idx_output, 1, AF_BINARY_ADD, false),
                      accum(idx_output, 1) - 1);
    idx_output.eval();
    // max size of bucket
    auto h = sum<unsigned int>(max(diff1(idx_output,0)).as(u64)) + 1;

    auto idx = batchFunc(idx_output(0,span), range(dim4(h, idx_output.dims(1)), 0, u64), BatchFunctions::batchAdd);
    auto idx_cpy = idx;
    idx_cpy(where(batchFunc(idx_cpy, idx_output(1,span), BatchFunctions::batchGreater))) = UINT64_MAX;
    {
        auto nonNullIdx = where(idx_cpy != UINT64_MAX);
        array nonNullData = idx_cpy(nonNullIdx);
        nonNullData = elements(nonNullData);
        nonNullData.eval();
        idx_cpy(nonNullIdx) = flipdims(nonNullData);
        idx_cpy.eval();
    }
    sort(idx_cpy, idx_output, idx_cpy, 0, isAscending);
    idx_output += range(idx_output.dims(), 1, idx_output.type()) * idx_output.dims(0);
    idx_output = idx_output(where(idx_cpy != UINT64_MAX));
    idx_output = idx(idx_output);
    idx_output.eval();
    return idx_output;
}

/* Currently does not work on strings longer than 8 characters */
void AFDataFrame::sortBy(int *columns, int size, bool const *isAscending) {
    bool asc = isAscending ? isAscending[0] : true;
    sortBy(columns[0], asc);
    for (int i = 1; i < size; ++i) {
        asc = isAscending ? isAscending[i] : true;
        array elements = hashColumn(columns[i], true);
        array buckets = hashColumn(columns[i - 1]);
        auto subsize = elements.dims(0);
        for (decltype(subsize) j = 0; j < subsize; ++j) {
            array idx = _subSort(elements(j, span), j ? elements(j - 1, span) : buckets, asc);
            _flush(idx);
        }
    }
}

array AFDataFrame::hashColumn(af::array const &column, DataType type, bool sortable) {
    if (type == STRING) return sortable ? prefixHash(column) : polyHash(prefixHash(column));
    if (type == DATE) return dateHash(column).as(u64);
    if (type == TIME) return timeHash(column).as(u64);
    if (type == DATETIME) return datetimeHash(column).as(u64);

    return array(column).as(u64);
}

AFDataFrame AFDataFrame::equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const {
    AFDataFrame result;

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

std::pair<array, array> AFDataFrame::crossCompare(af::array const &lhs, af::array const &rhs, batchFunc_t predicate) {
    auto r = where(batchFunc(flipdims(rhs), lhs, predicate));
    auto l = r / rhs.dims(1);
    r = r % rhs.dims(1);

    return { l, r };
}

std::pair<af::array, af::array> AFDataFrame::setCompare(array &lhs, array &rhs) {
    lhs = join(0, lhs, range(lhs.dims(), 1, u64));
    rhs = join(0, rhs, range(rhs.dims(), 1, u64));
    {
        array idx;
        array tmp;
        sort(tmp, idx, lhs.row(0), 1);
        lhs = lhs(span, idx);
        sort(tmp, idx, rhs.row(0), 1);
        rhs = rhs(span, idx);
    }

    auto setrl = flipdims(setIntersect(setUnique(lhs.row(0), true), setUnique(rhs.row(0), true), true));

    #if defined(AF_TEST)
        Logger::startTimer("AF");
        _removeNonExistant(setrl, lhs, rhs, 1);
        Logger::logTime("AF");
    #elif defined(USING_CUDA)
        Logger::startTimer("CUDA");
        _removeNonExistant(setrl, lhs, rhs);
        Logger::logTime("CUDA");
    #elif defined(USING_OPENCL)
        Logger::startTimer("OCL");
        _removeNonExistant(setrl, lhs, rhs);
        Logger::logTime("OCL");
    #else
        Logger::startTimer("CPU");
        _removeNonExistant(setrl, lhs, rhs);
        Logger::logTime("CPU");
    #endif

    auto cl = accum(join(1, constant(1, 1, u64), (diff1(lhs.row(0),1) > 0).as(u64)), 1) - 1;
    cl = flipdims(histogram(cl, cl.elements())).as(u64);
    cl = cl(cl > 0);
    auto il = scan(cl, 1, AF_BINARY_ADD, false);

    auto cr = accum(join(1, constant(1, 1, u64), (diff1(rhs.row(0),1) > 0).as(u64)), 1) - 1;
    cr = flipdims(histogram(cr, cr.elements())).as(u64);
    cr = cr(cr > 0);
    auto ir = scan(cr, 1, AF_BINARY_ADD, false);

    auto outpos = cr * cl;
    auto out_size = sum<unsigned int>(outpos);
    outpos = scan(outpos, 1, AF_BINARY_ADD, false);
    af::sync();

    auto x = setrl.elements();
    auto y = sum<unsigned int>(max(cl, 1));
    auto z = sum<unsigned int>(max(cr, 1));
    #if (!defined(AF_TEST) && defined(USING_CUDA))
    Logger::startTimer("CUDA Scatter");
    launch_IndexScatter(il, ir, cl, cr, outpos, l, r, x, y, z, out_size);
    Logger::logTime("CUDA Scatter");
    #else
    Logger::startTimer("AF Scatter");
    auto i = range(dim4(1, x * y * z), 1, u64);
    auto j = i / z % y;
    auto k = i % z;
    i = i / y / z;
    array l(1, out_size + 1, u64);
    array r(1, out_size + 1, u64);

    auto b = !(j / cl(i)) && !(k / cr(i));
    l(b * (outpos(i) + cl(i) * k + j) + !b * out_size) = il(i) + j;
    r(b * (outpos(i) + cr(i) * j + k) + !b * out_size) = ir(i) + k;
    Logger::logTime("AF Scatter");
    #endif

    l = l.cols(0,end - 1);
    r = r.cols(0,end - 1);
    lhs = lhs(1, l);
    rhs = rhs(1, r);
    lhs.eval();
    rhs.eval();
    return { lhs, rhs };
}

void AFDataFrame::_removeNonExistant(const array &setrl, array &lhs, array &rhs) {
    auto res_l = constant(0, dim4(1, lhs.row(0).elements() + 1), u64);
    auto res_r = constant(0, dim4(1, rhs.row(0).elements() + 1), u64);

    auto comp = setrl.device<ull>();
    auto result_left = res_l.device<ull>();
    auto result_right = res_r.device<ull>();
    auto input_l = lhs.device<ull>();
    auto input_r = rhs.device<ull>();
    af::sync();
    auto i_size = lhs.row(0).elements();
    printf("Thread count for lhs: %llu\n", i_size * setrl.elements());
    launch_IsExist(result_left, input_l, comp, i_size, setrl.elements());
    i_size = rhs.row(0).elements();
    printf("Thread count rhs: %llu\n", i_size * setrl.elements());
    launch_IsExist(result_right, input_r, comp, i_size, setrl.elements());

    setrl.unlock();
    lhs.unlock();
    rhs.unlock();
    res_l.unlock();
    res_r.unlock();

    res_r = res_r.cols(0, end - 1);
    res_l = res_l.cols(0, end - 1);
    lhs = lhs(span, where(res_l));
    rhs = rhs(span, where(res_r));
    lhs.eval();
    rhs.eval();
}

void AFDataFrame::_removeNonExistant(const array &setrl, array &lhs, array &rhs, bool swt) {
    auto res_l = constant(0, dim4(1, lhs.row(0).elements() + 1), u64);
    auto res_r = constant(0, dim4(1, rhs.row(0).elements() + 1), u64);

    auto i_size = lhs.row(0).elements();
    auto const comp_size = setrl.elements();
    printf("Thread count for lhs: %llu\n", i_size * comp_size);
    auto id = range(dim4(1, i_size * comp_size), 1, u64);
    auto i = id / comp_size;
    auto j = id % comp_size;
    auto b = moddims(setrl(j), i.dims()) == moddims(lhs(0, i),i.dims());
    auto k = b * i + !b * i_size;
    res_l(k) = 1;

    i_size = rhs.row(0).elements();
    printf("Thread count rhs: %llu\n", i_size * comp_size);
    id = range(dim4(1, i_size * comp_size), 1, u64);
    i = id / comp_size;
    j = id % comp_size;
    b = moddims(setrl(j), i.dims()) == moddims(rhs(0, i),i.dims());
    k = b * i + !b * i_size;
    res_r(k) = 1;

    res_r = res_r.cols(0, end - 1);
    res_l = res_l.cols(0, end - 1);
    lhs = lhs(span, where(res_l));
    rhs = rhs(span, where(res_r));
    lhs.eval();
    rhs.eval();
}

std::string AFDataFrame::name(std::string const& str) {
    _tableName = str;
    return _tableName;
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
        auto tmp = malloc(a.bytes());
        a.host(tmp);
        _hostData.emplace_back(tmp);
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
    _tableName.clear();
    _idxToName.clear();
    _nameToIdx.clear();
    if (!_hostData.empty()) {
        for (auto &dat : _hostData) freeHost(dat);
    }
    _hostData.clear();
    _dataTypes.clear();
}



