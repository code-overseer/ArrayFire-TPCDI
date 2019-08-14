#include "include/AFDataFrame.h"
#include "include/TPCDI_Utils.h"
#include "include/BatchFunctions.h"
#include "include/KernelInterface.h"
#include "include/Logger.h"
#ifndef ULL
#define ULL
    typedef unsigned long long ull;
#endif
using namespace BatchFunctions;
using namespace TPCDI_Utils;
using namespace af;

AFDataFrame::AFDataFrame(AFDataFrame&& other) noexcept : _data(std::move(other._data)),
                                                         _nameToCol(std::move(other._nameToCol)),
                                                         _colToName(std::move(other._colToName)),
                                                         _name(std::move(other._name)) {
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame&& other) noexcept {
    _data = std::move(other._data);
    _nameToCol = std::move(other._nameToCol);
    _colToName = std::move(other._colToName);
    _name = std::move(other._name);
    return *this;
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame const &other) noexcept {
    _data = other._data;
    _nameToCol = other._nameToCol;
    _name = other._name;
    return *this;
}

void AFDataFrame::add(Column &column, std::string const &name) {
    _data.emplace_back(column);
    if (!name.empty()) nameColumn(name, (int)(_data.size() - 1));
}

void AFDataFrame::add(Column &&column, std::string const &name) {
    _data.emplace_back(std::move(column));
    if (!name.empty()) nameColumn(name, (int)(_data.size() - 1));
}

void AFDataFrame::insert(Column &column, unsigned int index, std::string const &name) {
    _colToName.erase(index);
    for (auto &i : _nameToCol) {
        if (i.second >= index) {
            i.second += 1;
            _colToName.erase(i.second);
            _colToName.insert(std::make_pair(i.second, i.first));
        }
    }
    _data.insert(_data.begin() + index, column);
    if (!name.empty()) nameColumn(name, index);
}

void AFDataFrame::insert(Column &&column, unsigned int index, std::string const &name) { insert(column, index, name); }

void AFDataFrame::remove(unsigned int index) {
    _data.erase(_data.begin() + index);
    _nameToCol.erase(_colToName[index]);
    _colToName.erase(index);
    for (auto &i : _nameToCol) {
        if (i.second >= index) {
            i.second -= 1;
            _colToName.erase(i.second);
            _colToName.insert(std::make_pair(i.second, i.first));
        }
    }
}

AFDataFrame AFDataFrame::project(int const *columns, int size, std::string const &name) const {
    AFDataFrame output;
    output.name(name.empty() ? _name : name);
    for (int i = 0; i < size; i++) {
        int n = columns[i];
        output.add(Column(_data[n]));
        if (_colToName.count(n)) output.nameColumn(_colToName.at(n), i);
    }
    return output;
}

AFDataFrame AFDataFrame::select(af::array const &index, std::string const &name) const {
    AFDataFrame output;
    output.name(name.empty() ? _name : name);
    unsigned int i = 0;
    for (auto &a : _data) {
        if (_colToName.count(i)) {
            output.add(a.select(index), _colToName.at(i++));
        } else {
            output.add(a.select(index));
        }
    }
    return output;
}

AFDataFrame AFDataFrame::project(std::string const *columns, int size, std::string const &name) const {
    int order[size];
    for (int i = 0; i < size; i++) order[i] = _nameToCol.at(columns[i]);
    return project(order, size, name);
}

AFDataFrame AFDataFrame::project(str_list const columns, std::string const &name) const {
    return project(columns.begin(), columns.size(), name);
}

AFDataFrame AFDataFrame::zip(AFDataFrame const &rhs) const {
    if (rows() != rhs.rows()) throw std::runtime_error("Left and Right tables do not have the same length");
    AFDataFrame output = *this;

    for (size_t i = 0; i < rhs._data.size(); ++i)
        output.add(Column(rhs._data[i]), (rhs.name() + "." + rhs._colToName.at(i)));

    return output;
}

AFDataFrame AFDataFrame::unionize(AFDataFrame &frame) const {
    if (_data.size() != frame._data.size()) throw std::runtime_error("Number of attributes do not match");
    auto out(*this);
    for (size_t i = 0; i < out._data.size(); ++i)
        out._data[i] = out._data[i].concatenate(frame._data[i]);
    return out;
}

void AFDataFrame::sortBy(unsigned int const col, bool const isAscending) {
    array key = _data[col].hash(true);
    auto const size = key.dims(0);
    if (!size) return;
    array sorting;
    array idx;
    sort(sorting, idx, key(end, span), 1, isAscending);
    if ((int)size - 2 >= 0) {
        auto main = idx;
        for (int j = (int)size - 2; j >= 0; --j) {
            key = key(span, idx);
            sort(sorting, idx, key(j, span), 1, isAscending);
            main = main(idx);
        }
        for (auto &i : _data) i = i.select(main);
    } else {
        for (auto &i : _data) i = i.select(idx);
    }
    af::deviceGC();
}

void AFDataFrame::sortBy(unsigned int const *columns, unsigned int const size, bool const *isAscending) {
    for (int i = (int)size - 1; i >= 0; --i) {
        auto asc = isAscending ? isAscending[i] : true;
        sortBy(columns[i], asc);
    }
}

void AFDataFrame::sortBy(std::string const *columns, unsigned int const size, bool const *isAscending) {
    unsigned int order[size];
    for (int j = 0; j < size; ++j) order[j] = _nameToCol[columns[j]];
    sortBy(order, size, isAscending);
}

void AFDataFrame::sortBy(str_list const columns, bool_list const isAscending) {
    if (isAscending.size() && isAscending.size() != columns.size())
        throw std::runtime_error("column number and order type size do not match");
    sortBy(columns.begin(), columns.size(), isAscending.size() ? isAscending.begin() : nullptr);
}

AFDataFrame AFDataFrame::equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const {
    auto &left = _data[lhs_column];
    auto &right = rhs._data[rhs_column];
    if (left.type() != right.type()) throw std::runtime_error("Column type mismatch");
    if (left.isempty() || right.isempty()) return AFDataFrame();

    auto idx = setCompare(left.hash(), right.hash());

    if (idx.first.isempty()) return AFDataFrame();

    if (left.type() == STRING) {
        af::array l = left.index(af::span, idx.first);
        af::array r = right.index(af::span, idx.second);
        auto keep = stringComp(left.data(), right.data(), l, r);
        idx.first = idx.first(keep);
        idx.second = idx.second(keep);
    }

    AFDataFrame result;
    for (size_t i = 0; i < _data.size(); i++) {
        result.add(_data[i].select(idx.first), _colToName.at(i));
    }

    for (size_t i = 0; i < rhs._data.size(); i++) {
        if (i == rhs_column) continue;
        result.add(rhs._data[i].select(idx.second),
                   (rhs.name() + "." + rhs._colToName.at(i)));
    }

    return result;
}

std::pair<af::array, af::array> AFDataFrame::setCompare(Column const &lhs, Column const &rhs) {
    if (lhs.type() != rhs.type()) throw std::runtime_error("Column type mismatch");
    if (lhs.isempty() || rhs.isempty()) return { af::array(0, u64), af::array(0, u64) };
    if (lhs.type() == STRING || lhs.type() == TIME || lhs.type() == DATE || lhs.type() == DATETIME) {
        return setCompare(lhs.hash(), rhs.hash());
    }
    return setCompare(lhs.data(), rhs.data());
}

std::pair<af::array, af::array> AFDataFrame::setCompare(array const &left, array const &right) {
    if (left.isempty() || right.isempty()) return { af::array(0, u64), af::array(0, u64) };
    array lhs;
    array rhs;
    array idx;
    sort(lhs, idx, left, 1);
    lhs = join(0, lhs, idx.as(lhs.type()));
    sort(rhs, idx, right, 1);
    rhs = join(0, rhs, idx.as(rhs.type()));
    auto const equalSet = hflat(setIntersect(setUnique(lhs.row(0), true), setUnique(rhs.row(0), true), true));

    bagSetIntersect(lhs, equalSet);
    bagSetIntersect(rhs, equalSet);

    auto equals = equalSet.elements();
    joinScatter(lhs, rhs, equals);

    return { lhs, rhs };
}

std::string AFDataFrame::name(std::string const& str) {
    _name = str;
    return _name;
}

void AFDataFrame::nameColumn(std::string const& name, unsigned int column) {
    if (_colToName.count(column)) _nameToCol.erase(_colToName.at(column));
    _nameToCol[name] = column;
    _colToName[column] = name;
}

void AFDataFrame::flushToHost() {
    if (_data.empty()) return;
    for (auto &a : _data) a.toHost();
}

void AFDataFrame::clear() {
    _data.clear();
    _name.clear();
    _colToName.clear();
    _nameToCol.clear();
}


