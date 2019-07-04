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

AFParser::AFParser(char const *filename, char delimiter, bool hasHeader) : _filename(filename) {
    _length = 0;
    _width = 0;
    _maxColumnWidths = nullptr;
    _cumulativeMaxColumnWidths = nullptr;
    {
        std::string txt = loadFile(_filename);
        if (txt.back() != '\n') txt += '\n';
        _data = array(txt.size() + 1, txt.c_str()).as(u8);
        _data = _data(where(_data != '\r'));
    }
    _data.eval();
    _generateIndexer(delimiter, hasHeader);
    sync();
}

AFParser::AFParser(std::string const &text, char delimiter) {
    _length = 0;
    _width = 0;
    _maxColumnWidths = nullptr;
    _cumulativeMaxColumnWidths = nullptr;
    _data = array(text.size() + 1, text.c_str()).as(u8);
    _data = _data(where(_data != '\r'));
    _data.eval();
    _generateIndexer(delimiter, false);
    sync();
}

AFParser::~AFParser() {
    if (_maxColumnWidths) af::freeHost(_maxColumnWidths);
    if (_cumulativeMaxColumnWidths) af::freeHost(_cumulativeMaxColumnWidths);
}

std::string AFParser::loadFile(char const *filename) {
    std::ifstream file(filename);
    std::string text;
    file.seekg(0, std::ios::end);
    text.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
    return text;
}

void AFParser::_generateIndexer(char const delimiter, bool hasHeader) {
    _indexer = where(_data == '\n');
    _length = _indexer.elements();
    bool b;
    {
        auto col_end = where(_data == delimiter);
        _width = col_end.elements() / _length;
        col_end = moddims(col_end, _width++, _length);
        col_end = reorder(col_end, 1, 0);
        col_end.eval();
        _indexer = join(1, col_end, _indexer);
    }

    {
        auto row_start = constant(0, 1, u32);
        b = _indexer.dims(0) > 1;
        if (b) row_start = join(0, row_start, _indexer.col(end).rows(0, end - 1) + 1);

        row_start.eval();
        _indexer = join(1, row_start, _indexer);
    }

    if (hasHeader) {
        _indexer = b ? _indexer.rows(1, end) : array(0, 1, u32);
        _indexer.eval();
        --_length;
    }

    if (_maxColumnWidths) af::freeHost(_maxColumnWidths);
    if (_cumulativeMaxColumnWidths) af::freeHost(_cumulativeMaxColumnWidths);
    if (!_length) return;

    auto tmp = max(diff1(_indexer,1),0);
    tmp -= range(tmp.dims(), 1, u8).as(b8);
    _maxColumnWidths = tmp.host<uint32_t>();
    tmp = accum(tmp, 1) + range(tmp.dims(), 1, u32);
    _cumulativeMaxColumnWidths = tmp.host<uint32_t>();
}
/* This creates a copy of the column, TODO need to deal with missing values and 12/24hr format */
af::array AFParser::asTime(int column) const {
    if (!_length) return array(0, 3, u8);
    int8_t const i = column != 0;
    int8_t const len = 8;

    auto nulls = _indexer.col(column + 1) - _indexer.col(column) - i;
    nulls = where(nulls != len);
    nulls.eval();

    array out = batchFunc(range(dim4(1, len), 1, u32), _indexer.col(column) + i, batchAdd);
    out = moddims(_data(out), out.dims()) - '0';
    out(nulls,span) = 0;
    out(span,seq(2, 5, 3)) = 255;
    out = moddims(out(where(out >= 0 && out <= 9)), dim4(out.dims(0),6));
    out = batchFunc(out, flip(pow(10,range(dim4(1,6), 1, u32)),1), batchMul);
    out = sum(out, 1);
    out = join(1, out / 10000, out % 10000 / 100, out % 100).as(u8);
    out.eval();

    return out;
}

array AFParser::asDateTime(int const column, DateFormat const inputFormat) const {
    if (!_length) return array(0, 3, u16);
    int8_t const i = column != 0;
    int8_t const dlen = 10;
    int8_t const tlen = 8;
    int8_t const len = 19;
    auto const idx = _indexer.col(column) + i;

    auto nulls = _indexer.col(column + 1) - idx;
    nulls = where(nulls != len);
    nulls.eval();

    auto date = batchFunc(range(dim4(1, dlen), 1, u32), idx, batchAdd);

    date = moddims(_data(date), date.dims()) - '0';
    date(nulls,span) = 0;

    auto delims = _dateDelimIndices(inputFormat);
    date(span,seq(delims.first, delims.second, delims.second - delims.first)) = 255;
    date = moddims(date(where(date >= 0 && date <= 9)), dim4(date.dims(0),8));
    date = batchFunc(date, flip(pow(10,range(dim4(1,8), 1, u32)),1), batchMul);
    date = sum(date, 1);
    _dateKeyToDate(date, inputFormat);

    auto time = batchFunc(range(dim4(1, tlen), 1, u32) + dlen + 1, idx, batchAdd);
    time = moddims(_data(time), time.dims()) - '0';
    time(nulls,span) = 0;

    time(span,seq(2, 5, 3)) = 255;
    time = moddims(time(where(time >= 0 && time <= 9)), dim4(time.dims(0),6));
    time = batchFunc(time, flip(pow(10,range(dim4(1,6), 1, u32)),1), batchMul);
    time = sum(time, 1);
    time = join(1, time / 10000, time % 10000 / 100, time % 100).as(u16);
    time.eval();

    return join(1, date, time);
}

std::pair<int8_t, int8_t> AFParser::_dateDelimIndices(DateFormat const format) {
    if (format == YYYYDDMM || format == YYYYMMDD)
        return std::pair<int8_t, int8_t>(4, 7);

    return std::pair<int8_t, int8_t>(2, 5);
}

void AFParser::_dateKeyToDate(af::array &out, DateFormat const format) {
    switch (format) {
        case YYYYMMDD:
            out = join(1, out / 10000, out % 10000 / 100, out % 100).as(u16);
            return;
        case YYYYDDMM:
            out = join(1, out / 10000, out % 100, out % 10000 / 100).as(u16);
            return;
        case MMDDYYYY:
            out = join(1, out % 10000, out / 1000000, out % 10000 % 100).as(u16);
            return;
        case DDMMYYYY:
            out = join(1, out % 10000, out % 10000 % 100, out / 1000000).as(u16);
            return;
        default:
            throw std::runtime_error("No such date format");
    }
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asDate(int column, DateFormat inputFormat) const {
    if (!_length) return array(0, 3, u16);
    int8_t const i = column != 0;
    int8_t const len = 10;

    auto nulls = _indexer.col(column + 1) - _indexer.col(column) - i;
    nulls = where(nulls != len);
    nulls.eval();

    array out = batchFunc(range(dim4(1, len), 1, u32), _indexer.col(column) + i, batchAdd);
    out = moddims(_data(out), out.dims()) - '0';
    out(nulls,span) = 0;

    auto delims = _dateDelimIndices(inputFormat);
    out(span,seq(delims.first, delims.second, delims.second - delims.first)) = 255;
    out = moddims(out(where(out >= 0 && out <= 9)), dim4(out.dims(0),8));
    out = batchFunc(out, flip(pow(10,range(dim4(1,8), 1, u32)),1), batchMul);
    out = sum(out, 1);

    _dateKeyToDate(out, inputFormat);

    out.eval();

    return out;
}



/* generate chracter indices of a column invalid indices replaced with UINT32_MAX */
array AFParser::_generateReversedCharacterIndices(int const column) const {
    // Get the last character index
    unsigned int const i = column != 0;
    auto const maximum = _maxColumnWidths[column];
    auto out = _indexer.col(column + 1) - 1;
    // Get the indices of the whole number
    out = batchFunc(out, range(dim4(1, maximum), 1, u32), batchSub);
    // Removes the indices that do not point to part of the number (by pading these indices with UINT32_MAX)
    out(where(batchFunc(out, out.col(0), batchGreater))) = UINT32_MAX; // TODO may not need this line
    out(where(batchFunc(out, _indexer.col(column) + i, batchLess))) = UINT32_MAX;
    // Transpose then flatten the array so that it can be used to index _data
    out = flat(reorder(out,1,0));
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
    // set '-' to padded value
    output(cond) = 0;
    cond = where(output == '.');
    // cache decimal points
    {
        points = constant(-1, output.dims(0) / maximum, s32);
        points(cond / maximum) = (cond % maximum).as(s32);
    }
    output(cond) = 0;
    output(where(output == UINT32_MAX)) = 0;
    output(where(output)) -= '0';
    output = moddims(output, dim4(maximum, output.dims(0)/maximum));
    output = reorder(output,1,0);
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

array AFParser::asString(int column) const {
    if (!_length) return array(0, u8);
    unsigned int const i = column != 0;
    auto out = _indexer.col(column) + i;
    auto const maximum = _maxColumnWidths[column];
    if (!maximum) return constant(0, _length, u8);

    out = batchFunc(out, range(dim4(1, maximum + 1), 1, u32), batchAdd);
    out(where(batchFunc(out, _indexer.col(column + 1), batchGE))) = UINT32_MAX;
    out = flat(reorder(out,1,0));
    auto cond = where(out != UINT32_MAX);
    array tmp = out(cond);
    tmp = _data(tmp);
    tmp = tmp.as(u8);
    tmp.eval();
    out(where(out == UINT32_MAX)) = 0;
    out = out.as(u8);
    out(cond) = tmp;

    out = moddims(out, dim4(maximum + 1, out.elements()/(maximum + 1)));
    out = reorder(out, 1, 0);
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
    auto out = _indexer.col(column) + i;
    out = _data(out);
    out(where(out == 'T' || out == 't')) = 1;
    out(where(out != 1)) = 0;
    out.eval();
    return out;
}

af::array AFParser::_numParse(int column, af::dtype type) const {
    if (!_length) return array(0, type);
    auto const maximum = _maxColumnWidths[column];
    if (!maximum) return constant(0, _length, type);

    array negatives;
    array points;
    array out;
    _makeUniform(column, out, negatives, points);
    out = out.as(type);
    out.eval();
    {
        auto exponent = range(dim4(1, maximum), 1, s32);
        exponent = batchFunc(exponent, points, batchSub);
        exponent(where(exponent > 0)) -= 1;
        exponent = pow(10, exponent.as(type));
        out *= exponent;
    }
    out = sum(out, 1);
    if (!negatives.isempty()) out(negatives) *= -1;
    return out;
}



