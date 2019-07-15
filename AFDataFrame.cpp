//
// Created by Bryan Wong on 2019-06-27.
//

#include "AFDataFrame.h"
#include "BatchFunctions.h"
#include "Tests.h"
#include "Enums.h"

using namespace BatchFunctions;
using namespace af;
/* For debug */
void AFDataFrame::printStr(array str) {
    str.row(end) = '\n';
    str = str(where(str));
    str = join(0, flat(str), af::constant(0, 1, u8));
    str.eval();
    auto d = str.host<uint8_t>();
    print((char*)d);
    af::freeHost(d);
}

AFDataFrame::AFDataFrame(AFDataFrame&& other) noexcept : _deviceData(std::move(other._deviceData)),
                                                         _dataTypes(std::move(other._dataTypes)),
                                                         _length(other._length),
                                                         _nameToIdx(std::move(other._nameToIdx)),
                                                         _idxToName(std::move(other._idxToName)),
                                                         _tableName(std::move(other._tableName)) {
    if (!_length) {
        _rowIndexes = range(_deviceData[0].dims(), 0, u32);
        _rowIndexes.eval();
    }
    other._rowIndexes = array();
    other._length = 0;
}

AFDataFrame::AFDataFrame(AFDataFrame const &other) : _deviceData(other._deviceData),
                                                    _dataTypes(other._dataTypes),
                                                    _length(other._length),
                                                    _nameToIdx(other._nameToIdx),
                                                    _idxToName(other._idxToName),
                                                     _tableName(other._tableName + "_cpy") {
    if (!_length) {
        _rowIndexes = range(_deviceData[0].dims(), 0, u32);
        _rowIndexes.eval();
    }
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame&& other) noexcept {
    _deviceData = std::move(other._deviceData);
    _dataTypes = std::move(other._dataTypes);
    _length = other._length;
    if (!_length) {
        _rowIndexes = other._rowIndexes;
        _rowIndexes.eval();
    }
    _nameToIdx = std::move(other._nameToIdx);
    _idxToName = std::move(other._idxToName);
    _tableName = std::move(other._tableName);
    other._rowIndexes = array();
    other._length = 0;
    return *this;
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame const &other) noexcept {
    _deviceData = other._deviceData;
    _dataTypes = other._dataTypes;
    _nameToIdx = other._nameToIdx;
    _length = other._length;
    if (!_length) {
        _rowIndexes = range(_deviceData[0].dims(), 0, u32);
        _rowIndexes.eval();
    }
    _tableName = other._tableName + "_cpy";
    return *this;
}

void AFDataFrame::add(array &column, DataType type) {
    add(std::move(column), type);
}

void AFDataFrame::add(array &&column, DataType type) {
    _deviceData.emplace_back(column);
    _dataTypes.emplace_back(type);

    if (!_length) {
        _length = _deviceData[0].dims(1);
        _rowIndexes = range(dim4(1, _length), 1, u32);
    }
}

void AFDataFrame::insert(array &column, DataType type, int index) {
    insert(std::move(column), type, index);
}

void AFDataFrame::insert(array &&column, DataType type, int index) {
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

void AFDataFrame::_flush() {
    for (auto &i : _deviceData) {
        i = i(span, _rowIndexes);
        i.eval();
    }
    _length = _deviceData[0].dims(0);
    if (!_length) {
        _length = _deviceData[0].dims(0);
        _rowIndexes = range(dim4(1, _length), 1, u32);
    } else {
        _rowIndexes = array(0, u32);
    }
}

void AFDataFrame::stringLengthMatchSelect(int column, size_t length) {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Invalid column type");
    af::array& data = _deviceData[column];
    data = data.rows(0, length);
    _rowIndexes = where(data.row(end) == 0 && data.row(end - 1));
    _rowIndexes.eval();
    _flush();
}

void AFDataFrame::stringMatchSelect(int column, char const *str) {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Invalid column type");
    auto length = strlen(str);
    stringLengthMatchSelect(column, length);
    auto match = array(dim4(length + 1, 1), str); // include \0
    _rowIndexes = where(allTrue(batchFunc(_deviceData[column], match, batchEqual), 0));
    _rowIndexes.eval();
    _flush();
}

array AFDataFrame::dateOrTimeHash(int index) const {
    if (_dataTypes[index] != DATE) throw std::runtime_error("Invalid column type");
    return dateOrTimeHash(_deviceData[index]);
}

af::array AFDataFrame::dateOrTimeHash(array const &date) {
    auto mult = flip(pow(100, range(dim4(3,1), 0, u32)), 0);
    auto key = batchFunc(mult, date, batchMult);
    key = sum(key, 0);
    return key;
}

array AFDataFrame::datetimeHash(int index) const {
    if (_dataTypes[index] != DATETIME) throw std::runtime_error("Invalid column type");
    return datetimeHash(_deviceData[index]);
}

af::array AFDataFrame::datetimeHash(af::array const &datetime) {
    auto mult = flip(pow(100U, range(dim4(6,1), 0, u64)), 0);
    auto key = batchFunc(mult, datetime.as(u64), batchMult);
    key = sum(key, 0);
    return key;
}

void AFDataFrame::dateSort(int index, bool ascending) {
    auto keys = dateOrTimeHash(index);
    array out;
    sort(out, _rowIndexes, keys, 1, ascending);
    _flush();
}

af::array AFDataFrame::endDate() {
    return join(0, constant(9999, 1, u16), constant(12, 1, u16), constant(31, 1, u16));
}

af::array AFDataFrame::project(int column) const {
    return af::array(_deviceData[column]);
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

AFDataFrame AFDataFrame::project(std::string const *names, int size, std::string const &name) const {
    int columns[size];
    for (int i = 0; i < size; i++) {
        columns[i] = _nameToIdx.at(names[i]);
    }
    return project(columns, size, name);
}

void AFDataFrame::concatenate(AFDataFrame &&frame) {
    if (_dataTypes.size() != frame._dataTypes.size()) throw std::runtime_error("Number of attributes do not match");
    for (int i = 0; i < _dataTypes.size(); i++) {
        if (frame._dataTypes[i] != _dataTypes[i])
            throw std::runtime_error("Attribute types do not match");
    }

    for (int i = 0; i < _deviceData.size(); ++i) {
        af::array left;
        af::array right;
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

void AFDataFrame::concatenate(AFDataFrame &frame) {
    concatenate(std::move(frame));
}

bool AFDataFrame::isEmpty() {
    return _deviceData.empty();
}

af::array AFDataFrame::prefixHash(array const &column) {
    if (column.type() != u8) throw std::invalid_argument("Unexpected array type, input must be unsigned char");
    auto const n = column.dims(0);
    if (!n) throw std::runtime_error("Cannot hash null column");
    auto const r = n % 8;
    auto const h = n / 8 + 1 * (r != 0);
    auto const s1 = flip(range(dim4(8), 0, u64),0) * 8;
    auto out = sum(batchFunc(column.rows(0, 7), s1, bitShiftLeft), 0);
    for (int i = 1; i < h; ++i) {
        auto e = i * 8 + 7;
        if (e < n) out = join(0, out, sum(batchFunc(column.rows(i * 8, e), s1, bitShiftLeft), 0));
        else out = join(0, out, sum(batchFunc(column.rows(i * 8, i * 8 + r), s1.rows(0, r - 1), bitShiftLeft), 0));
    }

    return out;
}

af::array AFDataFrame::prefixHash(int column) const {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Expected string column");
    return prefixHash(_deviceData[column]);
}

af::array AFDataFrame::polyHash(array const &column) {
    uint64_t const prime = 71llU;
    auto hash = range(dim4(column.dims(0),1), 0, u64);
    hash = pow(prime, hash).as(u64);
    hash = batchFunc(column, hash, batchMult);
    hash = sum(hash, 0);
    hash.eval();
    return hash;
}

af::array AFDataFrame::polyHash(int column) const {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Expected string column");
    return polyHash(_deviceData[column]);
}

void AFDataFrame::sortBy(int column, bool isAscending) {

    array elements = hashColumn(column, true);
    auto const size = elements.dims(0);
    {
        array sorting;
        sort(sorting, _rowIndexes, elements(0, span), 1, isAscending);
    }
    _flush();
    for (int j = 1; j < size; ++j) {
        _rowIndexes = _subSort(elements(j, span), elements(j - 1, span), isAscending);
        _flush();
    }

}

array AFDataFrame::_subSort(array const &elements, array const &bucket, bool const isAscending) {
    array idx_output;
    {
        auto eq = sum(batchFunc(bucket, setUnique(bucket, isAscending), BatchFunctions::batchEqual),1);
        eq = flipdims(eq).as(u64);
        eq = join(1, constant(0,dim4(1), u64), eq);
        eq = accum(eq, 1);
        idx_output = join(0, eq.cols(0, end - 1), eq.cols(1, end) - 1);
    }
    // max size of bucket
    auto h = max(diff1(idx_output,0)).scalar<uint64_t>() + 1;
    auto idx = batchFunc(idx_output(0,span), range(dim4(h, idx_output.dims(1)), 0, u32), BatchFunctions::batchAdd);
    auto idx_cpy = idx;
    idx_cpy(where(batchFunc(idx_cpy, idx_output(1,span), BatchFunctions::batchGreater))) = UINT32_MAX;
    {
        auto nonNullIdx = where(idx_cpy!=UINT32_MAX);
        array nonNullData = idx_cpy(nonNullIdx);
        nonNullData = elements(nonNullData);
        idx_cpy(nonNullIdx) = flipdims(nonNullData);
    }
    sort(idx_cpy, idx_output, idx_cpy, 0, isAscending);
    idx_output += range(idx_output.dims(), 1, u32) * idx_output.dims(0);
    idx_output = idx_output(where(idx_cpy!=UINT32_MAX));
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
        for (int j = 0; j < subsize; ++j) {
            _rowIndexes = _subSort(elements(j, span), j ? elements(j - 1, span) : buckets, asc);
            _flush();
        }
    }
}

array AFDataFrame::hashColumn(af::array const &column, DataType type, bool sortable) {
    if (type == STRING) return sortable ? prefixHash(column) : polyHash(column);
    if (type == DATE || type == TIME) return dateOrTimeHash(column).as(u64);
    if (type == DATETIME) return datetimeHash(column).as(u64);

    return array(column).as(u64);
}

array AFDataFrame::hashColumn(int column, bool sortable) const {
    return hashColumn(_deviceData[column], _dataTypes[column], sortable);
}

AFDataFrame AFDataFrame::equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const {
    AFDataFrame result;
    af::array l;
    af::array r;
    auto &leftType = _dataTypes[lhs_column];
    auto &rightType = rhs._dataTypes[rhs_column];
    if (leftType != rightType) throw std::runtime_error("Supplied column data types do not match");

    l = hashColumn(lhs_column);
    r = rhs.hashColumn(rhs_column);
    l.eval();
    r.eval();

    auto idx = crossCompare(l, r);
    if (_deviceData[lhs_column].dims(0) <= rhs._deviceData[rhs_column].dims(0)) {
        l = _deviceData[lhs_column](span, idx.first);
        r = rhs._deviceData[rhs_column](range(dim4(l.dims(0)),0,u32), idx.second);
        r = moddims(r, l.dims());
    } else {
        r = rhs._deviceData[rhs_column](span, idx.second);
        l = _deviceData[lhs_column](range(dim4(r.dims(0)),0,u32), idx.first);
        l = moddims(l,r.dims());
    }
    l.eval();
    r.eval();
    {
        /* Collision Check */
        auto tmp = where(allTrue(l == r, 0));
        idx.first = idx.first(tmp);
        idx.second = idx.second(tmp);
    }

    for (int i = 0; i < _deviceData.size(); i++) {
        array tmp = _deviceData[i](span, idx.first);
        result.add(tmp, _dataTypes[i]);
        result.nameColumn(_idxToName.at(i), i);
    }

    for (int i = 0; i < rhs._deviceData.size(); i++) {
        if (i == rhs_column) continue;
        array tmp = rhs._deviceData[i](span, idx.second);
        result.add(tmp, rhs._dataTypes[i]);
        result.nameColumn(rhs.name() + "." + rhs._idxToName.at(i), result._deviceData.size() - 1);
    }

    return result;
}

std::pair<array, array> AFDataFrame::crossCompare(af::array const &lhs, af::array const &rhs, batchFunc_t predicate) {
    auto r = where(batchFunc(flipdims(rhs), lhs, predicate));

    auto l = r / rhs.dims(1);
    r = r % rhs.dims(1);

    return { l, r };
}

void AFDataFrame::reorder(int const *seq, int size) {
    if (size != _deviceData.size()) throw std::runtime_error("Invalid sequence size");
    int order[size];
    for (int i = 0; i < size; i++) order[seq[i]] = i;

    typedef typename std::iterator_traits<decltype(_dataTypes.begin())>::value_type type_t;
    typedef typename std::iterator_traits<decltype(_deviceData.begin())>::value_type value_t;
    typedef typename std::iterator_traits<decltype(seq)>::value_type index_t;
    typedef typename std::iterator_traits<decltype(seq)>::difference_type diff_t;


    auto v = _deviceData.begin();
    auto u = _dataTypes.begin();
    diff_t remaining = size - 1;
    for ( index_t s = index_t(), d; remaining > 0; ++s) {
        for ( d = order[s]; d > s; d = order[d] ) ;
        if ( d == s ) {
            --remaining;
            value_t tmp_arr = v[s];
            type_t tmp_type = u[s];
            while ( d = order[d], d != s ) {
                std::swap( tmp_arr, v[d] );
                std::swap( tmp_type, u[d] );
                --remaining;
            }
            v[s] = tmp_arr;
            u[s] = tmp_type;
        }
    }
}

std::string AFDataFrame::name() const {
    return _tableName;
}

std::string AFDataFrame::name(std::string const& str) {
    _tableName = str;
    return _tableName;
}

void AFDataFrame::reorder(std::string const *seq, int size)  {
    int seqnum[size];
    for (int j = 0; j < size; ++j)
        seqnum[j] = _nameToIdx[seq[j]];

    _nameToIdx.clear();
    _idxToName.clear();
    reorder(seqnum, size);
    for (int j = 0; j < size; ++j) {
        nameColumn(seq[j],j);
    }
}

void AFDataFrame::nameColumn(std::string const& name, int column) {
    if (_idxToName.count(column)) _nameToIdx.erase(_idxToName.at(column));
    _nameToIdx[name] = column;
    _idxToName[column] = name;
}

void AFDataFrame::nameColumn(std::string const& name, std::string const &old) {
    nameColumn(name, _nameToIdx.at(old));
}

// TODO move to finwireparser
array AFDataFrame::stringToDate(af::array &datestr, DateFormat inputFormat, bool isDelimited) {
    if (!datestr.dims(1)) return array(3, 0, u16);

    af::array out = datestr.rows(0, end - 1);
    out(where(out < '0' || out > '9')) = 0;
    auto nulls = where(allTrue(out == 0, 0));
    nulls.eval();
    out = out - '0';
    out(span, nulls) = 0;
    out = out(where(out >= 0 && out <= 9));
    out = moddims(out, dim4(8, out.dims(0)/8));
    out = batchFunc(out, flip(pow(10, range(dim4(8, 1), 0, u32)), 0), batchMult);
    out = sum(out, 0);

    AFParser::dateKeyToDate(out, inputFormat);

    out.eval();

    return out;
}



