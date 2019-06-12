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

using String = std::string;
template<typename T>
using Vector = std::vector<T>;
template<typename T>
using Predicate = std::function<bool(T)>;
typedef unsigned long ulong;
using namespace af;

array AFCSVParser::findChar(char c, array &csv) {
  auto r = constant((int)c, 1, csv.elements());
  r = csv - r;
  r = iszero(r);
  return where(r);
}

AFCSVParser AFCSVParser::parse(const char *filename, bool header) {
  AFCSVParser output;
  String txt = textToString(filename);
  
  output._data = array(1, txt.size() + 1, txt.c_str());
  auto row_end = findChar('\n', output._data);
  output._length = row_end.elements();
  auto row_start = shift(row_end + constant(1, output._length, u32), 1);
  row_start *= iszero(iszero(iota(output._length, 1, u32)));
  auto col_end = findChar(',', output._data);
  output._width = col_end.elements()/output._length;
  col_end = moddims(col_end, output._width, output._length);
  col_end = reorder(col_end, 1, 0);
  output._indexer = join(1,row_start, col_end, row_end);
  
  return output;
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
  for (int i = 0; i < _width; i++) {
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

void AFCSVParser::trim(array& selected_rows) {
  _length = selected_rows.elements();
  _indexer = _indexer(selected_rows,span);
}
