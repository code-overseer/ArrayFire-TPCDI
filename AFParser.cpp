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

AFParser::AFParser(char const *filename, char delimiter) {
    _length = 0;
    _width = 0;
    _maxColumnWidths = nullptr;
    _cumulativeMaxColumnWidths = nullptr;
    {
        std::string txt = loadFile(filename);
        _data = array(txt.size() + 1, txt.c_str()).as(u8);
        _data = _data(where(_data != '\r'));
    }
    _data.eval();
    _generateIndexer(delimiter);
    sync();
}

AFParser::AFParser(std::string const &text, char delimiter) {
    _length = 0;
    _width = 0;
    _maxColumnWidths = nullptr;
    _cumulativeMaxColumnWidths = nullptr;
    {
        _data = array(text.size() + 1, text.c_str()).as(u8);
        _data = _data(where(_data != '\r'));
    }
    _data.eval();
    _generateIndexer(delimiter);
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

void AFParser::_generateIndexer(char const delimiter) {
    _indexer = where(_data == '\n');
    _length = _indexer.elements();
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
        row_start = join(0, row_start, _indexer.col(end).rows(0, end - 1) + 1);
        row_start.eval();
        _indexer = join(1, row_start, _indexer);
    }
    _indexer.eval();
    if (_maxColumnWidths) af::freeHost(_maxColumnWidths);
    if (_cumulativeMaxColumnWidths) af::freeHost(_cumulativeMaxColumnWidths);
    auto tmp = max(diff1(_indexer,1),0);
    tmp -= range(tmp.dims(), 1, u8).as(b8);
    _maxColumnWidths = tmp.host<uint32_t>();
    tmp = accum(tmp, 1) + range(tmp.dims(), 1, u32);
    _cumulativeMaxColumnWidths = tmp.host<uint32_t>();
    tmp.col(end).scalar<uint32_t>();
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asDate(int column, DateFormat inputFormat, bool isDelimited) const {
    int8_t const offset = column != 0;
    int8_t const len = isDelimited ? 10 : 8;
    std::pair<int8_t, int8_t> year;
    std::pair<int8_t, int8_t> month;
    std::pair<int8_t, int8_t> day;

    switch (inputFormat) {
        case YYYYDDMM:
            year = std::pair<int8_t, int8_t>(0, 3);
            day = isDelimited ? std::pair<int8_t, int8_t>(5, 6) : std::pair<int8_t, int8_t>(4, 5);
            month = isDelimited ? std::pair<int8_t, int8_t>(8, 9) : std::pair<int8_t, int8_t>(7, 8);
            break;
        case YYYYMMDD:
            year = std::pair<int8_t, int8_t>(0, 3);
            month = isDelimited ? std::pair<int8_t, int8_t>(5, 6) : std::pair<int8_t, int8_t>(4, 5);
            day = isDelimited ? std::pair<int8_t, int8_t>(8, 9) : std::pair<int8_t, int8_t>(7, 8);
            break;
        case DDMMYYYY:
            year = isDelimited ? std::pair<int8_t, int8_t>(6, 9) : std::pair<int8_t, int8_t>(5, 8);
            day = std::pair<int8_t, int8_t>(0, 1);
            month = isDelimited ? std::pair<int8_t, int8_t>(3, 4) : std::pair<int8_t, int8_t>(2, 3);
            break;
        case MMDDYYYY:
            year = isDelimited ? std::pair<int8_t, int8_t>(6, 9) : std::pair<int8_t, int8_t>(5, 8);
            day = isDelimited ? std::pair<int8_t, int8_t>(3, 4) : std::pair<int8_t, int8_t>(2, 3);
            month = std::pair<int8_t, int8_t>(0, 1);
            break;
        default:
            throw std::invalid_argument("No such format!");
    }

    array out;
    {
        auto tmp = _indexer.col(column) + offset;
        out = batchFunc(range(dim4(1, len), 1, u32), tmp, batchAdd);
    }
    out = moddims(_data(out), out.dims()) - '0';
    out = moddims(out(where(out >= 0 && out <= 9)), dim4(out.dims(0),8));
    // matmul requires converting to f64 due to precision problems
    out = batchFunc(out, flip(pow(10,range(dim4(1,8), 1, u32)),1), batchMul);
    out = sum(out, 1);
    out = join(1, out / 10000, out % 10000 / 100, out % 100).as(u16);
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

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asUint(int const column) const {
    auto const maximum = _maxColumnWidths[column];
    array out;
    // Scoped to force memory to clear
    {
        array negatives;
        array points;
        _makeUniform(column, out, negatives, points);
    }

    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;
    // Multiply by powers of 10 and sum values across the row
    // Matmul can be used but requires casting between int and floats
    out = batchFunc(out, pow(10,range(dim4(1,maximum), 1, u32)), batchMul);
    out = sum(out, 1);
    out.eval();

    return out;
}

array AFParser::asInt(int const column) const {
    auto const maximum = _maxColumnWidths[column];
    array negatives;
    array out;
    // Scoped to force memory to clear
    {
        array points;
        _makeUniform(column, out, negatives, points);
    }
    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;

    out = batchFunc(out.as(s32), pow(10,range(dim4(1,maximum), 1, s32)), batchMul);
    out = sum(out, 1);
    // Negate to negative
    out(negatives) *= -1;
    out.eval();
    return out;
}

array AFParser::asFloat(int const column) const {
    auto const maximum = _maxColumnWidths[column];

    array negatives;
    array points;
    array out;
    _makeUniform(column, out, negatives, points);

    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out = out.as(f32);
    out.eval();
    {
        auto exponent = range(dim4(1, maximum), 1, s32);
        exponent = batchFunc(exponent, points.col(1), batchSub);
        exponent(where(exponent > 0)) -= 1;
        exponent = pow(10, exponent.as(f32));
        out *= exponent;
    }
    out = sum(out, 1);
    // Negate to negative
    out(negatives) *= -1;
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
    points = cond / maximum;
    points = join(1, points, (cond % maximum)).as(s32);
    output(cond) = 0;
    output(where(output == UINT32_MAX)) = 0;
    output(where(output)) -= '0';
}

array AFParser::asUlong(int const column) const {
    auto const maximum = _maxColumnWidths[column];
    array out;
    // Scoped to force memory to clear
    {
        array negatives;
        array points;
        _makeUniform(column, out, negatives, points);
    }

    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;

    // Multiply by powers of 10 and sum values across the row
    out = batchFunc(out.as(u64), pow(10,range(dim4(1,maximum), 1, s64)), batchMul);
    out = sum(out, 1);
    out.eval();

    return out;
}

array AFParser::asLong(int const column) const {
    auto const maximum = _maxColumnWidths[column];
    array negatives;
    array out;
    // Scoped to force memory to clear
    {
        array points;
        _makeUniform(column, out, negatives, points);
    }
    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;

    out = batchFunc(out.as(s64), pow(10,range(dim4(1,maximum), 1, s64)), batchMul);
    out = sum(out, 1);
    // Negate to negative
    out(negatives) *= -1;
    out.eval();
    return out;
}

array AFParser::asDouble(int column) const {
    auto const maximum = _maxColumnWidths[column];

    array negatives;
    array points;
    array out;
    _makeUniform(column, out, negatives, points);
    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out.eval();
    {
        auto exponent = range(dim4(1, maximum), 1, s32);
        exponent = batchFunc(exponent, points.col(1), batchSub);
        exponent(where(exponent > 0)) -= 1;
        exponent = pow(10, exponent.as(f64));
        out *= exponent;
    }
    out = sum(out, 1);
    // Negate to negative
    out(negatives) *= -1;
    out.eval();
    return out;
}

array AFParser::asString(int column) const {
    unsigned int const i = column != 0;
    auto out = _indexer.col(column) + i;
    auto const maximum = _maxColumnWidths[column];

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
    unsigned int const i = column != 0;
    auto out = _indexer.col(column) + i;
    out = _data(out);
    out(where(out == 'T' || out == 't')) = 1;
    out(where(out != 1)) = 0;
    out.eval();
    return out;
}
