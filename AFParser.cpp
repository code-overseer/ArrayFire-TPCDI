//
//  AFParser.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "AFParser.hpp"
#include <fstream>
#include "BatchFunctions.h"
#include "AFDataFrame.h"
#include <exception>
#include <sstream>
#include <utility>
#include <limits>

template<typename T>
using Vector = std::vector<T>;
template<typename T>
using Predicate = std::function<bool(T)>;
typedef unsigned long ulong;
using namespace af;
using namespace BatchFunctions;

AFParser::AFParser(char const *filename, char const delimiter, bool const hasHeader) : _filename(filename),
_length(0), _width(0), _maxColumnWidths(nullptr), _cumulativeMaxColumnWidths(nullptr) {
    {
        std::string txt = loadFile(_filename);
        if (txt.back() != '\n') txt += '\n';
        _data = array(txt.size() + 1, txt.c_str()).as(u8);
        _data = _data(where(_data != '\r'));
    }
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

std::string AFParser::loadFile(char const *filename) {
    std::ifstream file(filename);
    std::string text;
    file.seekg(0, std::ios::end);
    text.reserve(((size_t)file.tellg()) + 1);
    file.seekg(0, std::ios::beg);
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
    return text;
}

void AFParser::_colToRow(af::array& arr) {
    arr = moddims(arr, dim4(1, arr.elements()));
}

void AFParser::_generateIndexer(char const delimiter, bool hasHeader) {
    _indexer = where(_data == '\n');
    _colToRow(_indexer);
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
    if (!_length) return array(3, 0, u8);
    int8_t const i = column != 0;
    int8_t const len = 8;

    auto idx = _indexer.row(column) + i;
    auto nulls = _indexer.row(column + 1) - idx;
    nulls = where(nulls != len);
    nulls.eval();
    array out = batchFunc(range(dim4(len, 1), 0, u32), idx, batchAdd);

    out = moddims(_data(out), out.dims()) - '0';
    out(span, nulls) = 0;
    out(seq(2, 5, 3), span) = 255;
    out = moddims(out(where(out >= 0 && out <= 9)), dim4(6, out.dims(1)));

    out = batchFunc(out, flip(pow(10, range(dim4(6, 1), 0, u32)), 0), batchMult);
    out = sum(out, 0);
    out = join(0, out / 10000, out % 10000 / 100, out % 100).as(u8);
    out.eval();

    return out;
}

array AFParser::asDateTime(int const column, DateFormat const inputFormat) const {
    if (!_length) return array(6, 0, u16);
    int8_t const i = column != 0;
    int8_t const dlen = 10;
    int8_t const tlen = 8;
    int8_t const len = 19;
    auto const idx = _indexer.row(column) + i;

    auto nulls = _indexer.row(column + 1) - idx;
    nulls = where(nulls != len);
    nulls.eval();

    auto date = batchFunc(range(dim4(dlen, 1), 0, u32), idx, batchAdd);

    date = moddims(_data(date), date.dims()) - '0';
    date(span,nulls) = 0;

    auto delims = dateDelimIndices(inputFormat);
    date(seq(delims.first, delims.second, delims.second - delims.first), span) = 255;
    date = moddims(date(where(date >= 0 && date <= 9)), dim4(8, date.dims(1)));
    date = batchFunc(date, flip(pow(10, range(dim4(8, 1), 0, u32)), 0), batchMult);
    date = sum(date, 0);

    dateKeyToDate(date, inputFormat);

    auto time = batchFunc(range(dim4(tlen, 1), 0, u32) + dlen + 1, idx, batchAdd);
    time = moddims(_data(time), time.dims()) - '0';
    time(span, nulls) = 0;

    time(seq(2, 5, 3), span) = 255;
    time = moddims(time(where(time >= 0 && time <= 9)), dim4(6, time.dims(1)));

    time = batchFunc(time, flip(pow(10, range(dim4(6, 1), 0, u32)), 0), batchMult);
    time = sum(time, 0);
    time = join(0, time / 10000, time % 10000 / 100, time % 100).as(u16);
    time.eval();

    return join(0, date, time);
}

std::pair<int8_t, int8_t> AFParser::dateDelimIndices(DateFormat format) {
    if (format == YYYYDDMM || format == YYYYMMDD)
        return { 4, 7 };

    return { 2, 5 };
}

void AFParser::dateKeyToDate(af::array &out, DateFormat format) {
    switch (format) {
        case YYYYMMDD:
            out = join(0, out / 10000, out % 10000 / 100, out % 100).as(u16);
            return;
        case YYYYDDMM:
            out = join(0, out / 10000, out % 100, out % 10000 / 100).as(u16);
            return;
        case MMDDYYYY:
            out = join(0, out % 10000, out / 1000000, out % 10000 % 100).as(u16);
            return;
        case DDMMYYYY:
            out = join(0, out % 10000, out % 10000 % 100, out / 1000000).as(u16);
            return;
        default:
            throw std::runtime_error("No such date format");
    }
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asDate(int column, DateFormat inputFormat, bool isDelimited) const {
    if (!_length) return array(3, 0, u16);
    int8_t const i = column != 0;
    int8_t const len = isDelimited ? 10 : 8;

    auto idx = _indexer.row(column) + i;

    auto nulls = _indexer.row(column + 1) - idx;
    nulls = where(nulls != len);
    nulls.eval();

    array out = batchFunc(range(dim4(len, 1), 0, u32), idx, batchAdd);

    out = moddims(_data(out), out.dims()) - '0';
    out(span, nulls) = 0;

    if (isDelimited) {
        auto delims = dateDelimIndices(inputFormat);
        out(seq(delims.first, delims.second, delims.second - delims.first), span) = UINT8_MAX;
    }

    out = moddims(out(where(out >= 0 && out <= 9)), dim4(8, out.dims(1)));
    out = batchFunc(out, flip(pow(10, range(dim4(8, 1), 0, u32)), 0), batchMult);
    out = sum(out, 0);

    dateKeyToDate(out, inputFormat);

    out.eval();

    return out;
}

/* generate chracter indices of a column invalid indices replaced with UINT32_MAX */
array AFParser::_generateReversedCharacterIndices(int const column) const {
    // Get the last character index
    unsigned int const i = column != 0;
    auto const maximum = _maxColumnWidths[column];
    auto out = _indexer.row(column + 1) - 1;
    // Get the indices of the whole number
    out = batchFunc(out, range(dim4(maximum, 1), 0, u32), batchSub);
    // Removes the indices that do not point to part of the number (by pading these indices with UINT32_MAX)
    out(where(batchFunc(out, out.row(0), batchGreater))) = UINT32_MAX; // TODO may not need this line
    out(where(batchFunc(out, _indexer.row(column) + i, batchLess))) = UINT32_MAX;
    // Transpose then flatten the array so that it can be used to index _data
    out = flat(out);
    out.eval();

    return out;
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

void AFParser::_makeUniform(int column, array &output, array &negatives, array &points) const {
    auto const maximum = _maxColumnWidths[column];
    output = _generateReversedCharacterIndices(column);
    auto cond = where(output != UINT32_MAX);
    {
        array tmp = output(cond);
        tmp = _data(tmp);
        tmp = tmp.as(u32);
        tmp.eval();
        output(cond) = tmp;
    }

    cond = where(output == '-');
    // cache negative rows
    negatives = cond / maximum;
    negatives = moddims(negatives, dim4(1, negatives.dims(0)));
    // set '-' to padded value
    output(cond) = 0;

    cond = where(output == '.');
    // cache decimal points
    points = constant(-1, 1, output.dims(0) / maximum, s32);
    points(cond / maximum) = (cond % maximum).as(s32);


    output(cond) = 0;
    output(where(output == UINT32_MAX)) = 0;
    output(where(output)) -= '0';
    output = moddims(output, dim4(maximum, output.dims(0)/maximum));
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
af::array AFParser::stringToBoolean(int column) const {
    if (!_length) return array(0, b8);
    unsigned int const i = column != 0;
    auto out = _indexer.row(column) + i;
    out = _data(out);
    out(where(out == 'T' || out == 't' || out == '1')) = 1;
    out(where(out != 1)) = 0;
    out = moddims(out, dim4(out.dims(1), out.dims(0)));
    out.eval();
    return out;
}

af::array AFParser::_numParse(int column, af::dtype type) const {
    if (!_length) return array(0, type);
    auto const maximum = _maxColumnWidths[column];
    if (!maximum) return constant(0, 1, _length, type);

    array negatives;
    array points;
    array out;
    _makeUniform(column, out, negatives, points);

    out = out.as(type);
    out.eval();
    {
        auto exponent = range(dim4(maximum, 1), 0, s32);
        exponent = batchFunc(exponent, points, batchSub);
        exponent(where(exponent > 0)) -= 1;
        exponent = pow(10, exponent.as(type));
        out *= exponent;
    }
    out = sum(out, 0);

    if (!negatives.isempty()) out(negatives) *= -1;
    return out;
}



