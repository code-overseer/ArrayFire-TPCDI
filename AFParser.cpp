//
//  AFParser.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "AFParser.hpp"
#include "BatchFunctions.h"
#include "TPCDI_Utils.h"
#include <sstream>
#include <utility>

using namespace af;
using namespace BatchFunctions;
using namespace TPCDI_Utils;

AFParser::AFParser(char const *filename, char const delimiter, bool const hasHeader) : _filename(filename),
_length(0), _width(0), _maxColumnWidths(nullptr), _cumulativeMaxColumnWidths(nullptr) {
    {
        std::string txt = TPCDI_Utils::loadFile(_filename);
        if (txt.back() != '\n') txt += '\n';
        _data = array(txt.size() + 1, txt.c_str()).as(u8);
        _data = _data(where(_data != '\r'));
    }
    _data.eval();
    _generateIndexer(delimiter, hasHeader);
    af::sync();
}

AFParser::AFParser(const std::vector<std::string> &files, char const delimiter, bool const hasHeader) : _filename(nullptr),
_length(0), _width(0), _maxColumnWidths(nullptr), _cumulativeMaxColumnWidths(nullptr) {
    auto text = collect(files);
    _data = array(text.size() + 1, text.c_str()).as(u8);
    _data = _data(where(_data != '\r'));
    _data.eval();
    _generateIndexer(delimiter, hasHeader);
    af::sync();
}

AFParser::AFParser(std::string const &text, char const delimiter, bool const hasHeader) : _filename(nullptr),
_length(0), _width(0), _maxColumnWidths(nullptr), _cumulativeMaxColumnWidths(nullptr){
    _data = array(text.size() + 1, text.c_str()).as(u8);
    _data = _data(where(_data != '\r'));
    _data.eval();
    _generateIndexer(delimiter, hasHeader);
    af::sync();
}

AFParser::~AFParser() {
    if (_maxColumnWidths) af::freeHost(_maxColumnWidths);
    if (_cumulativeMaxColumnWidths) af::freeHost(_cumulativeMaxColumnWidths);
}

void AFParser::_generateIndexer(char const delimiter, bool hasHeader) {
    _indexer = where(_data == '\n');
    _indexer = flipdims(_indexer);
    _length = _indexer.elements();
    {
        auto col_end = where(_data == delimiter);
        _width = col_end.elements() / _length;
        col_end = moddims(col_end, _width++, _length);
        col_end.eval();
        _indexer = join(0, col_end, _indexer);
    }

    if (!_indexer.isempty()) {
        auto row_start = constant(0, 1, u32);
        if (_length > 1) row_start = join(1, row_start, _indexer.row(end).cols(0, end - 1) + 1);
        row_start.eval();
        _indexer = join(0, row_start, _indexer);
    }

    if (hasHeader) {
        _indexer = _indexer.dims(1) <= 1 ? array(1, 0, u32) : _indexer.cols(1, end);
        _indexer.eval();
        --_length;
    }

    if (_maxColumnWidths) af::freeHost(_maxColumnWidths);
    if (_cumulativeMaxColumnWidths) af::freeHost(_cumulativeMaxColumnWidths);
    if (!_length) return;

    auto tmp = max(diff1(_indexer,0),1);
    tmp -= range(tmp.dims(), 0, u8) > 0;
    _maxColumnWidths = tmp.host<uint32_t>();
    tmp = accum(tmp, 0) + range(tmp.dims(), 0, u32);
    _cumulativeMaxColumnWidths = tmp.host<uint32_t>();
}

af::array AFParser::asTime(int column) const {
    auto out = _makeUniform(column);
    return stringToTime(out, true);
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asDate(int column, bool isDelimited, DateFormat inputFormat) const {
    auto out = _makeUniform(column);
    return stringToDate(out, isDelimited, inputFormat);
}

af::array AFParser::asDateTime(int column, bool isDelimited, DateFormat inputFormat) const {
    auto out = _makeUniform(column);
    return stringToDateTime(out, isDelimited, inputFormat);
}

array AFParser::asUchar(int column) const {
    auto out = _numParse(column, u8);
    out.eval();
    return out;
}

array AFParser::asUshort(int column) const {
    auto out = _numParse(column, u16);
    out.eval();
    return out;
}

array AFParser::asShort(int column) const {
    auto out = _numParse(column, s16);
    out.eval();
    return out;
}

array AFParser::asUint(int const column) const {
    auto out = _numParse(column, u32);
    out.eval();
    return out;
}

array AFParser::asInt(int const column) const {
    auto out = _numParse(column, s32);
    out.eval();
    return out;
}

array AFParser::asFloat(int const column) const {
    auto out = _numParse(column, f32);
    out.eval();
    return out;
}

array AFParser::asU64(int const column) const {
    auto out = _numParse(column, u64);
    out.eval();
    return out;
}

array AFParser::asS64(int const column) const {
    auto out = _numParse(column, s64);
    out.eval();
    return out;
}

array AFParser::asDouble(int const column) const {
    auto out = _numParse(column, f64);
    out.eval();
    return out;
}

af::array AFParser::_makeUniform(int column) const {
    auto const maximum = _maxColumnWidths[column] + 1; // delimiter
    unsigned int const i = column != 0;
    auto out = static_cast<array>(_indexer.row(column + 1));
    // Get the indices of the whole number
    out = batchFunc(out, range(dim4(maximum), 0, u32), batchSub);
    // Removes the indices that do not point to part of the number (by pading these indices with UINT32_MAX)

    out(where(batchFunc(out, _indexer.row(column + 1), batchGreater))) = UINT32_MAX;
    out(where(batchFunc(out, _indexer.row(column) + i, batchLess))) = UINT32_MAX;
    // Transpose then flatten the array so that it can be used to index _data

    out = flip(out, 0);
    auto cond = where(out != UINT32_MAX);
    out(cond) = _data(static_cast<array>(out(cond))).as(u32);
    out(out == UINT32_MAX) = ' ';
    out = moddims(out, dim4(maximum, out.elements() / maximum)).as(u8);
    out.row(end) = 0;
    out.eval();

    return out;
}

array AFParser::asString(int column) const {
    if (!_length) return array(0, u8);
    unsigned int const i = column != 0;
    auto out = _indexer.row(column) + i;
    auto const maximum = _maxColumnWidths[column];
    if (!maximum) return constant(0, 1, _length, u8);

    out = batchFunc(out, range(dim4(maximum + 1, 1), 0, u32), batchAdd);
    out(where(batchFunc(out, _indexer.row(column + 1), batchGE))) = UINT32_MAX;
    out = flat(out);
    auto cond = where(out != UINT32_MAX);
    array tmp = out(cond);
    tmp = _data(tmp);
    tmp = tmp.as(u8);
    tmp.eval();
    out(where(out == UINT32_MAX)) = 0;
    out = out.as(u8);
    out(cond) = tmp;

    out = moddims(out, dim4(maximum + 1, out.elements()/(maximum + 1)));
    out.eval();

    return out;
}

void AFParser::printData() const {
    auto c = _data.host<uint8_t>();
    print((char*)c);
    freeHost(c);
}

// TODO add nulls
af::array AFParser::asBoolean(int column) const {
    if (!_length) return array(0, b8);
    unsigned int const i = column != 0;
    auto out = _indexer.row(column) + i;
    out = _data(out);
    return stringToBoolean(out);
}

af::array AFParser::_numParse(int column, af::dtype type) const {
    auto out = _makeUniform(column);
    return stringToNum(out, type);
}



