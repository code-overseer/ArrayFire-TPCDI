//
//  AFCSVParser.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "AFCSVParser.hpp"
#include "Utils.hpp"
#include <exception>
#include <sstream>

using String = std::string;
template<typename T>
using Vector = std::vector<T>;
template<typename T>
using Predicate = std::function<bool(T)>;
typedef unsigned long ulong;
using namespace af;

array AFCSVParser::findChar(char c, array &csv) {
  return where(csv == (int)c);
}

AFCSVParser AFCSVParser::parse(const char *filename, bool header) {
  AFCSVParser output;
  {
    String txt = textToString(filename);
    output._data = array(txt.size() + 1, txt.c_str());
    output._data.eval();
  }
  output._generateIndexer();
  return output;
}

void AFCSVParser::_generateIndexer() {
  auto row_end = findChar('\n', _data);
  _length = row_end.elements();
  auto row_start = shift(row_end + constant(1, _length, u32), 1);
  row_start *= iota(_length, 1, u32) != 0;
  auto col_end = findChar(',', _data);
  
  _width = col_end.elements()/_length;
  col_end = moddims(col_end, _width, _length);
  col_end = reorder(col_end, 1, 0);
  _indexer = join(1,row_start, col_end, row_end);
  _indexer.eval();
}

String AFCSVParser::get(dim_t row, dim_t col) const {
  auto idx = _indexer.host<unsigned int>();
  if (row > _length) throw std::invalid_argument("row index exceeds length");
  if (col > _width) throw std::invalid_argument("col index exceeds width");
  auto start = idx[col * _length + row] + !!col;
  auto length = idx[(col + 1) * _length + row] - start;
  freeHost(idx);
  return _getString().substr(start, length);
}

void AFCSVParser::printRow(std::ostream& str, unsigned long row) const {
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
void AFCSVParser::printColumn(std::ostream& str, unsigned long col) const {
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

String AFCSVParser::_getString() const {
  auto str = _data.host<char>();
  auto out = String(str);
  freeHost(str);
  return out;
}

void AFCSVParser::select(unsigned int col, char const* match) {
  unsigned int const len = (unsigned int)strlen(match);
  unsigned int const cond = !!col;
  
  auto starts = _indexer.col(col) + cond;
  auto i = _indexer.col(col + 1) - starts;
  auto out = i == len;
  out = where(out);
  // out contains the row indices where the string length matches the target string
  starts = starts(out);
  // starts contains the data indices where the string length matches the target string
  i = tile(range(dim4(1,len),1,u32), starts.dims());
  // i contains the character index of the target string
  
  auto str = tile(array(1, len, match), starts.dims());
  i = tile(starts, 1, len) + i;
  i = moddims(_data(i), i.dims()) - str;
  i = where(!anyTrue(i,1));
  i = out(i);
  
  _length = i.elements();
  _indexer = _indexer(i, span);
  _indexer.eval();
  
  i = max(_indexer.col(end) - _indexer.col(0)).as(u32);
  array c = _indexer.col(0);
  auto l = i.scalar<unsigned int>() + 1;
  sync();
  auto add = range(dim4(_indexer.col(0).dims(0), l), 1, u32);
  auto lims = tile(_indexer.col(end), 1 , l);
  c = tile(c, 1, l) + add;
  c(where(c > lims)) = 0;
  c = flat(reorder(c,1,0));
  c = c(where(c));
  c.eval();
  _data = join(0,_data(c),constant(0,1,_data.type()));
  _data.eval();
  
  _generateIndexer();
}
