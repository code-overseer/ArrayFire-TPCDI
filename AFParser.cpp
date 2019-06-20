//
//  AFParser.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "AFParser.hpp"
#include "Utils.hpp"
#include <exception>
#include <sstream>
#include <utility>

using String = std::string;
template<typename T>
using Vector = std::vector<T>;
template<typename T>
using Predicate = std::function<bool(T)>;
typedef unsigned long ulong;
using namespace af;

array AFParser::findChar(char c, array const &txt) {
    return where(txt == (int)c);
}

AFParser AFParser::parse(char const *filename, char const delimiter) {
    array data;
    String txt;
    loadFile(filename, txt);
    return AFParser(txt, delimiter);
}

void AFParser::loadFile(char const *filename, std::string &text) {
    std::ifstream file(filename);

    file.seekg(0, std::ios::end);
    text.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
}

void AFParser::_generateIndexer() {

    _indexer = findChar('\n', _data);
    _length = _indexer.elements();

    {
        auto col_end = findChar(_delimiter, _data);
        _width = col_end.elements() / _length;
        col_end = moddims(col_end, _width, _length);
        col_end = reorder(col_end, 1, 0);
        _indexer = join(1, col_end, _indexer);
    }

    {
        auto row_start = constant(0, 1, u32);
        row_start = join(0, row_start, _indexer.col(end).rows(0, end - 1) + 1);
        _indexer = join(1, row_start, _indexer);
    }
    _indexer.eval();
}

array AFParser::dateGenerator(uint16_t d, uint16_t m, uint16_t y) {
//    char       buf[64];
//    if (!d || !m || !y) {
//        time_t     now = time(0);
//        struct tm  time{};
//        time = *localtime(&now);
//        strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &time);
//        String date(buf);
//        date.erase(date.begin() + 10, date.end());
//        return array(1, date.size(), date.c_str());
//    }
//    sprintf(buf, "%d-%d-%d", y, m, d);
//    String date(buf);
    int date [] = {y, m, d};
    return array(1, 3, date);
}

AFParser::AFParser(String const &txt, char const delimiter) {
    _length = 0;
    _width = 0;
    _delimiter = delimiter;
    _data = array(txt.size() + 1, txt.c_str());
    _data.eval();
    _generateIndexer();
    sync();
}

void AFParser::asDate(int const column, af::array &out, DateFormat inputFormat, bool isDelimited) const {
    int8_t const cond = column != 0;
    int8_t const len = isDelimited ? 10 : 8;
    std::pair<int8_t, int8_t> year;
    std::pair<int8_t, int8_t> month;
    std::pair<int8_t, int8_t> day;

    switch (inputFormat) {
        case YYYYDDMM:
            year = std::pair<int8_t, int8_t>(0, 3);
            month = isDelimited ? std::pair<int8_t, int8_t>(5, 6) : std::pair<int8_t, int8_t>(4, 5);
            day = isDelimited ? std::pair<int8_t, int8_t>(8, 9) : std::pair<int8_t, int8_t>(7, 8);
            break;
        case YYYYMMDD:
            year = std::pair<int8_t, int8_t>(0, 3);
            day = isDelimited ? std::pair<int8_t, int8_t>(5, 6) : std::pair<int8_t, int8_t>(4, 5);
            month = isDelimited ? std::pair<int8_t, int8_t>(8, 9) : std::pair<int8_t, int8_t>(7, 8);
            break;
        case DDMMYYYY:
            year = isDelimited ? std::pair<int8_t, int8_t>(6, 9) : std::pair<int8_t, int8_t>(5, 8);
            day = isDelimited ? std::pair<int8_t, int8_t>(3, 4) : std::pair<int8_t, int8_t>(2, 3);
            month = std::pair<int8_t, int8_t>(0, 1);
            break;
        case MMDDYYYY:
            break;
        default:
            throw std::invalid_argument("No such format!");
    }

    array inter;

    auto start = _indexer.col(column) + cond;
    inter = tile(range(dim4(1, len), 1, u32), start.dims());
    inter = tile(start, 1, len) + inter;
    inter = moddims(_data(inter), inter.dims()) - '0';
    /* Year */
    auto lhs = inter(span,seq(year.first, year.second)).as(f32);
    auto rhs = flip(pow(10,range(dim4(4,1),0, f32)),0);
    lhs = matmul(lhs, rhs);
    out = lhs;
    /* Month */
    lhs = inter(span, seq(month.first, month.second)).as(f32);
    rhs = flip(pow(10,range(dim4(2,1),0, f32)),0);
    lhs = matmul(lhs, rhs);
    out = join(1, out, lhs);
    /* Day */
    lhs = inter(span, seq(day.first, day.second)).as(f32);
    lhs = matmul(lhs, rhs);
    out = join(1, out, lhs);

    out = out.as(u16);
    out.eval();
}

void AFParser::asDate(std::string column, af::array &out, DateFormat inputFormat, bool isDelimited) {
    asDate(_columnNames[column], out, inputFormat, isDelimited);
}

String AFParser::get(dim_t row, dim_t col) const {
    auto idx = _indexer.host<unsigned int>();
    if (row > _length) throw std::invalid_argument("row index exceeds length");
    if (col > _width) throw std::invalid_argument("col index exceeds width");
    auto start = idx[col * _length + row] + !!col;
    auto length = idx[(col + 1) * _length + row] - start;
    freeHost(idx);
    return _getString().substr(start, length);
}
void AFParser::printRow(std::ostream& str, unsigned long row) const {
    auto idx = _indexer.host<unsigned int>();
    if (row > _length) throw std::invalid_argument("row index exceeds length");
    ulong start;
    ulong length;
    auto dat = _getString();
    for (unsigned int i = 0; i < _width; i++) {
        start = idx[i * _length + row] + !!i;
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
        start = idx[col * _length + i] + !!col;
        length = idx[(col + 1) * _length + i] - start;
        str << dat.substr(start, length);
        if (i < _length - 1) str << '\n';
    }
    str << std::endl;
    freeHost(idx);
}

String AFParser::_getString() const {
    auto str = _data.host<char>();
    auto out = String(str);
    freeHost(str);
    return out;
}

void AFParser::select(unsigned int col, char const* match) {
    unsigned int const len = (unsigned int)strlen(match);
    unsigned int const cond = !!col;

    auto starts = _indexer.col(col) + cond;
    auto i = _indexer.col(col + 1) - starts;
    auto out = i == len; // Tests whether length matches the desired string length
    out = where(out); // Gets row indices where length matches
    starts = starts(out); // get the character array indices
    i = tile(range(dim4(1,len),1,u32), starts.dims()); // generate array of {0,1,2,...len-1},{0,1,2,...len-1},...

    auto str = tile(array(1, len, match), starts.dims()); // tile 'match' {'match'},{'match'},{'match'}

    i = tile(starts, 1, len) + i; // Generate positions of characters in '_data' to be compared
    i = allTrue(moddims(_data(i), i.dims()) == str, 1); // test for equality with match
    i = where(i); // get row indices where the string matches
    i = out(i);

    _length = i.elements();
    _indexer = _indexer(i, span);
    _indexer.eval();

    i = max(_indexer.col(end) - _indexer.col(0)).as(u32);
    array c = _indexer.col(0);
    auto l = i.scalar<unsigned int>() + 1; /* Inefficient */
    sync();
    auto add = range(dim4(_indexer.col(0).dims(0), l), 1, u32);
    auto limits = tile(_indexer.col(end), 1 , l);
    c = tile(c, 1, l) + add;
    c(where(c > limits)) = 0;
    c = flat(reorder(c,1,0));
    c = c(where(c));
    c.eval();
    _data = join(0,_data(c),constant(0,1,_data.type()));
    _data.eval();

    _generateIndexer();
    sync();
}

void AFParser::nameColumn(std::string const name, unsigned long const idx) {
    _columnNames[name] = idx;
}

void AFParser::nameColumn(std::string const name, std::string const oldName) {
    _columnNames[name] = _columnNames[oldName];
    _columnNames.erase(oldName);
}