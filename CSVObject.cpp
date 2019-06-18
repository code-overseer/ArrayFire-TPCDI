//
//  CSVObject.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "CSVObject.hpp"
#include "Utils.hpp"

using String = std::string;
template<typename T>
using Vector = std::vector<T>;
template<typename T>
using Predicate = std::function<bool(T)>;

CSVObject CSVObject::parse(const char *filename, bool header) {
  String csv = textToString(filename);
  CSVObject output;
  auto data = output._data;
  auto rows = _findRowSlice(csv);
  
  for (Slice r: rows) {
    auto cols = _findColSlice(csv, r);
    for (ulong c_idx = 0; c_idx < cols.size(); c_idx++) {
      auto c = cols[c_idx];
      if (c_idx == data->size()) data->push_back(Vector<String>());
      (*data)[c_idx].push_back(csv.substr(r.start, r.length).substr(c.start, c.length));
    }
  }
  return output;
}

CSVObject::CSVObject(CSVObject& src, Vector<ulong>& selected) {
  auto toCopy = src._data;
  _columnNames = src._columnNames;
  
  for (int i = 0; i < (*toCopy).size(); i++) {
    _data->push_back(Vector<String>());
    for (auto j : selected) {
      (*_data)[i].push_back((*toCopy)[i][j]);
    }
  }
}

Vector<Slice> SliceCSV(char c, String const &csv) {
  ulong start = 0;
  ulong end = 0;
  std::vector<Slice> r;
  for (auto i = csv.begin(); i != csv.end(); i++, end++) {
    if (c == *i) {
      r.push_back(Slice(start, end));
      start = end + 1;
    }
  }
  return r;
}

Vector<Slice> CSVObject::_findRowSlice(String const &csv) {
  return SliceCSV('\n', csv);
}

Vector<Slice> CSVObject::_findColSlice(String const &csv, Slice const row) {
  return SliceCSV(',', csv.substr(row.start, row.length));
}

std::vector<ulong> CSVObject::select(int column, Predicate<String> func) const{
  std::vector<ulong> selected_rows;
  ulong sz = (*_data)[column].size();
  for (ulong i = 0; i < sz; i++) {
    if (func((*_data)[column][i])) selected_rows.push_back(i);
  }
  std::sort(selected_rows.begin(), selected_rows.end());
  return selected_rows;
}

void CSVObject::printRow(std::ostream& str, ulong row) const{
  int i = 0;
  for (; i < _data->size() - 1; i++) {
    str << (*_data)[i][row] << ',';
  }
  str<<(*_data)[i][row]<<std::endl;
}

void CSVObject::printColumn(std::ostream& str, ulong col) const{
  for (auto r: (*_data)[col]) {
    str << r << '\n';
  }
  str<<std::endl;
}

String CSVObject::get(ulong row, ulong col) const{
  return (*_data)[row][col];
}

bool CSVObject::nameColumn(String name, ulong idx) {
  return _columnNames.insert(std::make_pair(name, idx)).second;
}

bool CSVObject::nameColumn(String name, String old) {
  if (!_columnNames.count(old)) return false;
  auto idx = _columnNames[old];
  auto result = _columnNames.insert(std::make_pair(name, idx)).second;
  if (!result) return false;
  _columnNames.erase(old);
  return true;
}

void CSVObject::trim(Vector<ulong>& selected) {
  ulong k = 0;
  for (ulong j = 0; j < length() && k < selected.size(); j++) {
    if (j == selected[k]) {
      for (int i = 0; i < width(); i++) {
        std::swap((*_data)[i][j], (*_data)[i][k]);
      }
      k++;
    }
  }
  for (ulong j = length() - 1; j >= k; j--) {
    for (int i = 0; i < width(); i++) {
      (*_data)[i].pop_back();
    }
  }
}

ulong CSVObject::length() const{
  return (*_data)[0].size();
}

ulong CSVObject::width() const {
  return (*_data).size();
}
