//
// Created by Bryan Wong on 2019-06-27.
//

#include "AFDataFrame.h"
#include "BatchFunctions.h"
#include "Tests.h"
#include <thread>
using namespace BatchFunctions;
using namespace af;
/* For debug */
void AFDataFrame::printStr(array &str) {
    auto a = str;
    a.col(end) = '\n';
    a = reorder(a, 1, 0);
    a = a(where(a));
    a = join(0, a, af::constant(0, 1, u8));
    a.eval();
    auto d = a.host<uint8_t>();
    print((char*)d);
    af::freeHost(d);
}

AFDataFrame::AFDataFrame(AFDataFrame&& other) noexcept : _length(0) {
    _deviceData = std::move(other._deviceData);
    _dataTypes = std::move(other._dataTypes);
    _length = other._length;
    if (!_length) {
        _rowIndexes = range(_deviceData[0].dims(), 0, u32);
        _rowIndexes.eval();
    }
    _columnNames = std::move(other._columnNames);
    other._length = 0;
}

void AFDataFrame::add(array &column, DataType type) {
    _deviceData.emplace_back(column);
    _dataTypes.push_back(type);

    if (!_length) {
        _length = _deviceData[0].dims(0);
        _rowIndexes = range(_deviceData[0].dims(), 0, u32);
    }
}

void AFDataFrame::insert(array &column, DataType type, int index) {
    _deviceData.insert(_deviceData.begin() + index, column);
    _dataTypes.insert(_dataTypes.begin() + index, type);
}

void AFDataFrame::remove(int index) {
    _deviceData.erase(_deviceData.begin() + index);
    _dataTypes.erase(_dataTypes.begin() + index);
}

std::vector<array> &AFDataFrame::data() {
    return _deviceData;
}

void AFDataFrame::_flush() {
    for (auto i = _deviceData.begin(); i != _deviceData.end(); i++) {
        *i = (*i)(_rowIndexes, span);
        i->eval();
    }
    _length = _deviceData[0].dims(0);
    if (!_length) {
        _rowIndexes = range(_deviceData[0].dims(), 0, u32);
        _rowIndexes.eval();
    } else {
        _rowIndexes = array().as(u32);
    }
}

af::array AFDataFrame::_generateStringIndex(int column) {
    auto eol = where(_deviceData[column] == '\n').as(u32);
    auto starts = join(0, constant(0,1, u32), eol.rows(0, end - 1) + 1);
    array idx = join(1, starts, eol);
    idx.eval();
    return idx;
}

void AFDataFrame::stringLengthMatch(int column, size_t length) {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Invalid column type");
    af::array& data = _deviceData[column];
    data = data.cols(0, length);
    _rowIndexes = where(data.col(end) == 0 && data.col(end - 1));
    _rowIndexes.eval();
    _flush();
}

void AFDataFrame::stringMatch(int column, char const* str) {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Invalid column type");
    auto length = strlen(str);
    stringLengthMatch(column, length);
    auto match = array(dim4(1, length + 1), str); // include \0
    _rowIndexes = where(allTrue(batchFunc(_deviceData[column], match, batchEqual), 1));
    _rowIndexes.eval();
    _flush();
}

array AFDataFrame::dateKeyGen(int index) {
    if (_dataTypes[index] != DATE) throw std::runtime_error("Invalid column type");
    af::array& col = _deviceData[index];
    auto mult = flip(pow(100, range(dim4(1,3), 1, u32)), 1);
    auto key = batchFunc(mult, col, batchMul);
    key = sum(key, 1);
    return key;
}

void AFDataFrame::dateSort(int index, bool ascending) {
    auto keys = dateKeyGen(index);
    array out;
    array idx;
    sort(out, _rowIndexes, keys, 0, ascending);
    _flush();
}

af::array AFDataFrame::endDate() {
    return join(1, constant(9999, 1, u16), constant(12, 1, u16), constant(31, 1, u16));
}

void AFDataFrame::concatenate(AFDataFrame &frame) {
    if (_dataTypes.size() != frame._dataTypes.size()) throw std::runtime_error("Number of attributes do not match");
    for (int i = 0; i < _dataTypes.size(); i++) {
        if (frame._dataTypes[i] != _dataTypes[i])
            throw std::runtime_error("Attribute types do not match");
    }

    for (int i = 0; i < _deviceData.size(); ++i) {
        if (_dataTypes[i] == STRING) {
            auto delta = _deviceData[i].dims(1) - frame._deviceData[i].dims(1);
            if (delta > 0) {
                auto back = constant(0, frame._deviceData[i].dims(0), delta, u8);
                frame._deviceData[i] = join(1, frame._deviceData[i], back);
            } else if (delta < 0) {
                delta = -delta;
                auto back = constant(0, _deviceData[i].dims(0), delta, u8);
                _deviceData[i] = join(1, _deviceData[i], back);
            }
        }
        _deviceData[i] = join(0, _deviceData[i], frame._deviceData[i]);
    }
}

bool AFDataFrame::isEmpty() {
    return _deviceData.empty();
}

