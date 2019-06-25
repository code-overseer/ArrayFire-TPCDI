//
//  AFParser.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "AFParser.hpp"
#include "Utils.hpp"
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
    _maxRowWidth = 0;
    _maxColumnWidths = nullptr;
    _cumulativeMaxColumnWidths = nullptr;
    _delimiter = delimiter;

    {
        std::string txt = loadFile(filename);
        _data = array(txt.size() + 1, txt.c_str()).as(u8);
        auto tmp = where(_data != '\r');
        if (!tmp.isempty()) _data = _data(tmp);
    }
    _data.eval();
    _generateIndexer();
    sync();
}

AFParser::~AFParser() {
    if (_maxColumnWidths) af::freeHost(_maxColumnWidths);
    if (_cumulativeMaxColumnWidths) af::freeHost(_cumulativeMaxColumnWidths);
}

array AFParser::findChar(char c, array const &txt) {
    return where(txt == (int)c);
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

void AFParser::_generateIndexer() {
    _data = join(0, _data, constant(0,1, u8));
    _data.eval();
    _indexer = findChar('\n', _data);
    _length = _indexer.elements();
    {
        auto col_end = findChar(_delimiter, _data);
        _width = col_end.elements() / _length;
        col_end = moddims(col_end, _width++, _length);
        col_end = reorder(col_end, 1, 0);
        _indexer = join(1, col_end, _indexer);
    }

    {
        auto row_start = constant(0, 1, u32);
        row_start = join(0, row_start, _indexer.col(end).rows(0, end - 1) + 1);
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
    _maxRowWidth = tmp.col(end).scalar<uint32_t>();
}

array AFParser::dateGenerator(uint16_t d, uint16_t m, uint16_t y) {
    array date(1, u32);
    date = y * 10000 + m * 100 + d;
    return date;
}

array AFParser::endDate() {
    array date(1, u32);
    date = 9999 * 10000 + 12 * 100 + 31;
    return date;
}

array AFParser::serializeDate(af::array const &date) {
    array delim(1, u8);
    delim = '-';
    auto tmp = batchFunc(date, flip(pow(10,range(dim4(1,8), 1, u32)),1), batchDiv);
    tmp = (tmp % 10).as(u8);
    tmp += '0';
    auto out = join(1, tmp.cols(0,3), delim, tmp.cols(4,5), delim);
    out = join(1, out, tmp.cols(6,7), constant('\n', out.dims(0), u8));
    out.eval();
    return out;
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asDate(int const column, DateFormat inputFormat, bool isDelimited) const {
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
    out.eval();

    return out;
}

/* This creates a copy of the column */
array AFParser::asDate(std::string column, DateFormat inputFormat, bool isDelimited) {
    return asDate(_columnNames[column], inputFormat, isDelimited);
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
    out(where(batchFunc(out, out.col(0), batchGreater))) = UINT32_MAX;
    out(where(batchFunc(out, _indexer.col(column) + i, batchLess))) = UINT32_MAX;
    // Transpose then flatten the array so that it can be used to index _data
    out = flat(reorder(out,1,0));
    out.eval();
    return out;
}

/* This creates a copy of the column, TODO need to deal with missing values */
array AFParser::asUnsigned32(int const column) {
    auto const maximum = _maxColumnWidths[column];

    auto out = _generateReversedCharacterIndices(column);
    // Scoped to force memory to clear
    {
        auto cond1 = where(out != UINT32_MAX);
        array j = out(cond1);
        j = _data(j);
        j = (j - '0').as(u32);
        j.eval();
        out(cond1) = j;
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

array AFParser::asUnsigned32(std::string column) {
    return asUnsigned32(_columnNames[column]);
}

array AFParser::asSigned32(int const column) {
    auto const maximum = _maxColumnWidths[column];
    auto out = _generateReversedCharacterIndices(column);

    array negatives;
    // Scoped to force memory to clear
    {
        auto cond = where(out != UINT32_MAX);
        array tmp = out(cond);
        tmp = _data(tmp);
        tmp = (tmp - '0').as(u32);
        tmp.eval();
        out(cond) = tmp;
        cond = where(out == '-' - '0');
        // cache negative rows
        negatives = cond / maximum;
        // set '-' to padded value
        out(cond) = 0;
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

array AFParser::asSigned32(std::string const column) {
    return asSigned32(_columnNames[column]);
}

array AFParser::asFloat(int const column) {
    unsigned int const i = column != 0;
    auto const maximum = _maxColumnWidths[column];
    auto out = _generateReversedCharacterIndices(column);
    array negatives;
    array points;

    // Scoped to force memory to clear
    {
        auto cond = where(out != UINT32_MAX);
        array tmp = out(cond);
        tmp = _data(tmp);
        tmp = (tmp - '0').as(u32);
        tmp.eval();
        out(cond) = tmp;
        cond = where(out == '-' - '0');
        // cache negative rows
        negatives = cond / maximum;
        // set '-' to padded value
        out(cond) = 0;
        cond = where(out == '.' - '0');
        // cache decimal points
        points = cond/maximum;
        points = join(1, points, (cond % maximum)).as(s32);
        out(cond) = 0;
    }
    // Reverse the transpose and flatten done before
    out = moddims(out, dim4(maximum, out.dims(0)/maximum));
    out = reorder(out,1,0);
    // Set the padded values to 0
    out(where(out == UINT32_MAX)) = 0;
    out = out.as(f32);
    out.eval();
    {
        auto exponent = range(dim4(1, maximum), 1, s32);
        exponent = batchFunc(exponent, points.col(1), batchSub);
        exponent(where(exponent > 0)) -= 1;
        exponent = pow(10, exponent.as(f32));
        out *= exponent;
    }
//    out = batchFunc(out, pow(10,range(dim4(1,maximum), 1, s32)), batchMul);
    out = sum(out, 1);
    // Negate to negative
    out(negatives) *= -1;
    out.eval();

    return out;
}

array AFParser::asFloat(std::string const column){
    return asFloat(_columnNames[column]);
}

void AFParser::printRow(std::ostream& str, unsigned long row) const {
    auto idx = _indexer.host<unsigned int>();
    if (row > _length) throw std::invalid_argument("row index exceeds length");
    ulong start;
    ulong length;
    auto dat = _getString();
    for (unsigned int i = 0; i < _width; i++) {
        start = idx[i * _length + row] + (i != 0);
        length = idx[(i + 1) * _length + row] - start;
        str << dat.substr(start, length);
        if (i < _width - 1) str << ' ';
    }
    str << std::endl;
    freeHost(idx);
}

void AFParser::printColumn(std::ostream& str, unsigned long col) const {
    auto idx = _indexer.host<unsigned int>();
    if (col > _width) throw std::invalid_argument("col index exceeds width");
    ulong start;
    ulong length;
    auto dat = _getString();
    for (int i = 0; i < _length; i++) {
        start = idx[col * _length + i] + (col != 0);
        length = idx[(col + 1) * _length + i] - start;
        str << dat.substr(start, length);
        if (i < _length - 1) str << '\n';
    }
    str << std::endl;
    freeHost(idx);
}

void AFParser::printData() {
    std::cout<<_getString()<<std::endl;
}

af::array AFParser::serializeUnsignedInt(af::array &integer) {
    array out;
    array pads;
    {
        auto tmp = log10(integer).as(u32);
        out = max(tmp);
        pads = batchFunc(out, tmp, batchSub);
    }
    auto n = out.scalar<unsigned int>(); // Known to be inefficient
    {
        auto tmp = range(dim4(1, n + 1), 1, u32);
        pads = batchFunc(tmp , pads, batchLess);
        out = batchFunc(integer, flip(pow(10, tmp),1), batchDiv);
    }
    out = out % 10;
    out += '0';
    out(where(pads)) = 0;
    out = join(1, out.as(u8), constant('\n',out.dims(0), u8));
    out = flat(reorder(out, 1, 0));
    out = out(where(out != 0));
    out.eval();
    return out;
}

void AFParser::insertAsFirst(af::array &serializedInput) {
    insertInto(0, serializedInput);
}

void AFParser::insertAsLast(af::array &input) {
    insertInto(_width, input);
}

array AFParser::_prepareColumnForInsert(af::array input, bool toComma) const {

    auto colEnd = findChar('\n', input);
    if (colEnd.isempty()) throw std::invalid_argument("No newline charcter found in input");
    if (toComma) input(colEnd) = ',';
    if (input.dims(1) != 1) return input;
    auto colStart = join(0, constant(0, dim4(1), u32), colEnd.rows(0, end - 1) + 1);
    auto length = colEnd - colStart + 1;
    colStart = batchFunc(colStart,range(dim4(1, max(length).scalar<uint32_t>()), 1, u32), batchAdd);
    auto pads = batchFunc(colStart,colEnd, batchGreater);
    input = moddims(input(colStart), colStart.dims());
    input(pads) = 0;
    return input;
}

void AFParser::insertInto(int column, af::array &serializedInput) {
    if (column < 0) throw std::invalid_argument("Indices must be > 0");
    if (column > _width) throw std::invalid_argument("Indices must be < number of columns");

    serializedInput = _prepareColumnForInsert(serializedInput, column != _width);
    auto left = _getLeftOf(column);
    auto right = _getRightOf(column);
    {
        auto tmp = join(1, left, serializedInput);
        tmp = join(1, tmp, right);
        tmp.eval();
        _data = flat(reorder(tmp, 1, 0));
        _data = _data(where(_data != 0));
        _data.eval();
        _generateIndexer();
    }

}

void AFParser::removeColumn(int column) {
    if (column < 0) throw std::invalid_argument("Indices must be > 0");
    if (column > _width) throw std::invalid_argument("Indices must be < number of columns");
    array left = _getLeftOf(column);
    array right = _getRightOf(column + 1);
    {
        auto tmp = join(1, left, right);
        _data = flat(reorder(tmp, 1, 0));
        _data = _data(where(_data != 0));
        _data.eval();
        _generateIndexer();
    }
}

array AFParser::_getRightOf(int column) const {
    if (column >= _width) return array(_indexer.dims(0), 0, u8);
    auto const i = (uint32_t) (column != 0);
    auto const j = ((column == 0) ? -1 : _cumulativeMaxColumnWidths[column - 1]);

    auto rhs = _maxRowWidth - j;
    auto right = range(dim4(1, rhs), 1, u32);
    right = batchFunc(right, _indexer.col(column) + i, batchAdd);
    {
        auto rightDim = right.dims();
        auto rightPad = where(batchFunc(right, _indexer.col(end), batchGreater));
        right = moddims(_data(right), rightDim);
        right(rightPad) = 0;
        right.eval();
    }
    return right;
}

array AFParser::_getLeftOf(int column) const {
    if (column <= 0) return array(_indexer.dims(0), 0, u8);
    auto const i = column == _width;
    auto lhs = _cumulativeMaxColumnWidths[column - 1] + 1;
    auto left = range(dim4(1, lhs), 1, u32);
    left = batchFunc(left, _indexer.col(0), batchAdd);
    {
        auto leftDim = left.dims();
        auto leftPad = where(batchFunc(left, _indexer.col(!i * column + i * end), batchGreater));
        left = moddims(_data(left), leftDim);
        left(leftPad) = 0;
        left(findChar('\n', left)) = ',';
        left.eval();
    }
    return left;
}

std::string AFParser::_getString() const {
    auto str = _data.host<uint8_t >();
    auto out = std::string((char*)str);
    freeHost(str);
    return out;
}

/* Returns rows that match */
array AFParser::stringMatch(unsigned int col, char const *match) {
    auto const len = (unsigned int)strlen(match);
    unsigned int const offset = col != 0;

    auto starts = _indexer.col(col) + offset;
    auto i = _indexer.col(col + 1) - starts;
    auto out = i == len; // Tests whether length matches the desired string length
    out = where(out); // Gets row indices where length matches
    out.eval();
    starts = starts(out); // get the character array indices
    starts.eval();
    i = batchFunc(starts, range(dim4(1, len), 1, u32), batchAdd);

    i = allTrue(batchFunc(moddims(_data(i), i.dims()), array(1, len, match), batchEqual), 1);
    i = where(i); // get row indices where the string matches
    i = out(i);
    i.eval();
    return i;
}

void AFParser::keepRows(af::array rows) {
    {
        auto tmp = batchFunc(range(dim4(1, _maxRowWidth), 1, u32), _indexer(rows, 0), batchAdd);
        tmp(where(batchFunc(tmp, _indexer(rows, end), batchGreater))) = UINT32_MAX;
        tmp = flat(reorder(tmp, 1, 0));
        tmp = tmp(where(tmp != UINT32_MAX));
        tmp.eval();
        _data = _data(tmp);
        _data.eval();
    }
    _generateIndexer();
}

void AFParser::removeRows(af::array rows) {

}

void AFParser::nameColumn(std::string const name, unsigned long const idx) {
    _columnNames[name] = idx;
}

void AFParser::nameColumn(std::string const name, std::string const oldName) {
    _columnNames[name] = _columnNames[oldName];
    _columnNames.erase(oldName);
}