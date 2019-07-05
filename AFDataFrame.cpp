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
void AFDataFrame::printStr(array str) {
    str.row(end) = '\n';
    str = str(where(str));
    str = join(0, flat(str), af::constant(0, 1, u8));
    str.eval();
    auto d = str.host<uint8_t>();
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
    _dataTypes.emplace_back(type);

    if (!_length) {
        _length = _deviceData[0].dims(0);
        _rowIndexes = range(dim4(1, _length), 1, u32);
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
        *i = (*i)(span, _rowIndexes);
        i->eval();
    }
    _length = _deviceData[0].dims(0);
    if (!_length) {
        _length = _deviceData[0].dims(0);
        _rowIndexes = range(dim4(1, _length), 1, u32);
    } else {
        _rowIndexes = array(0, u32);
    }
}

void AFDataFrame::stringLengthMatch(int column, size_t length) {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Invalid column type");
    af::array& data = _deviceData[column];
    data = data.rows(0, length);
    _rowIndexes = where(data.row(end) == 0 && data.row(end - 1));
    _rowIndexes.eval();
    _flush();
}

void AFDataFrame::stringMatch(int column, char const* str) {
    if (_dataTypes[column] != STRING) throw std::runtime_error("Invalid column type");
    auto length = strlen(str);
    stringLengthMatch(column, length);
    auto match = array(dim4(length + 1, 1), str); // include \0
    _rowIndexes = where(allTrue(batchFunc(_deviceData[column], match, batchEqual), 0));
    _rowIndexes.eval();
    _flush();
}

array AFDataFrame::dateKeyGen(int index) {
    if (_dataTypes[index] != DATE) throw std::runtime_error("Invalid column type");
    af::array& col = _deviceData[index];
    auto mult = flip(pow(100, range(dim4(3,1), 0, u32)), 0);
    auto key = batchFunc(mult, col, batchMul);
    key = sum(key, 0);
    return key;
}

void AFDataFrame::dateSort(int index, bool ascending) {
    auto keys = dateKeyGen(index);
    array out;
    sort(out, _rowIndexes, keys, 1, ascending);
    _flush();
}

af::array AFDataFrame::endDate() {
    return join(0, constant(9999, 1, u16), constant(12, 1, u16), constant(31, 1, u16));
}

void AFDataFrame::concatenate(AFDataFrame &frame) {
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

bool AFDataFrame::isEmpty() {
    return _deviceData.empty();
}

