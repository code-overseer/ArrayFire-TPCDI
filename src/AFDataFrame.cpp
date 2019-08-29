#include "AFDataFrame.h"
#include "BatchFunctions.h"
#include "KernelInterface.h"
#include "AFHashTable.h"
#include "Utils.h"
#include "Logger.h"

typedef unsigned long long ull;

using namespace BatchFunctions;
using namespace Utils;
using namespace af;

AFDataFrame::AFDataFrame(AFDataFrame&& other) noexcept : _columns(std::move(other._columns)),
                                                         _nameToCol(std::move(other._nameToCol)),
                                                         _colToName(std::move(other._colToName)),
                                                         _name(std::move(other._name)) {
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame&& other) noexcept {
    _columns = std::move(other._columns);
    _nameToCol = std::move(other._nameToCol);
    _colToName = std::move(other._colToName);
    _name = std::move(other._name);
    return *this;
}

AFDataFrame& AFDataFrame::operator=(AFDataFrame const &other) noexcept {
    _columns = other._columns;
    _nameToCol = other._nameToCol;
    _name = other._name;
    return *this;
}

void AFDataFrame::add(Column &column, std::string const &name) {
    _columns.emplace_back(column);
    if (!name.empty()) nameColumn(name, (int)(_columns.size() - 1));
}

void AFDataFrame::add(Column &&column, std::string const &name) {
    _columns.emplace_back(std::move(column));
    if (!name.empty()) nameColumn(name, (int)(_columns.size() - 1));
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
    _columns.insert(_columns.begin() + index, column);
    if (!name.empty()) nameColumn(name, index);
}

void AFDataFrame::insert(Column &&column, unsigned int index, std::string const &name) { insert(column, index, name); }

void AFDataFrame::remove(unsigned int index) {
    _columns.erase(_columns.begin() + index);
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
        output.add(Column(_columns[n]));
        if (_colToName.count(n)) output.nameColumn(_colToName.at(n), i);
    }
    return output;
}

AFDataFrame AFDataFrame::select(af::array const &index, std::string const &name) const {
    AFDataFrame output;
    output.name(name.empty() ? _name : name);
    unsigned int i = 0;
    for (auto &a : _columns) {
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

    for (size_t i = 0; i < rhs._columns.size(); ++i)
        output.add(Column(rhs._columns[i]), (rhs.name() + "." + rhs._colToName.at(i)));

    return output;
}

AFDataFrame AFDataFrame::sum(std::string const &col, str_list group_by) const {
    AFDataFrame output;
    auto const &agg = _columns[_nameToCol.at(col)];
    if (agg.type() == STRING || agg.type() == DATE || agg.type() == TIME || agg.type() == DATETIME) {
        throw std::runtime_error("Expected numeric type to aggregate");
    }
    if (!group_by.size()) {
        output.add(Column(af::sum(agg.data(), 1)), "SUM("+col+")");
        return output;
    }
    af::array group_key = _columns[_nameToCol.at(*group_by.begin())].hash();
    int j = 0;
    for (auto i = group_by.begin() + 1; i != group_by.end(); ++i, j = !j) {
        if (!j) group_key = group_key ^ (_columns[_nameToCol.at(*i)].hash() << 2);
        else group_key = group_key ^ (_columns[_nameToCol.at(*i)].hash() >> 2);
    }
    auto to_sum(agg.data());
    af::array output_group_idx;
    sort(group_key, output_group_idx, group_key, 1);
    to_sum = to_sum(output_group_idx);

    auto diffe = diff1(group_key, 1) > 0;
    auto indexer = where64(join(1, af::constant(1,1,diffe.type()), diffe));
    auto key_count = hflat(where64(join(1, diffe, af::constant(1,1,diffe.type()))) - indexer + 1); // histogram

    auto loops = af::sum<unsigned long long>(af::max(key_count, 1));
    auto summation = af::constant(0, key_count.dims(), to_sum.type());
    for (ull i = 0; i < loops; ++i) {
        auto b = i < key_count;
        summation(b) += (to_sum(indexer(b) + i) + 0);
    }

    output.add(Column(summation), "SUM("+col+")");
    for (const auto & i : group_by) {
        output.add(_columns[_nameToCol.at(i)].select(output_group_idx), i);
    }

    return output;
}

AFDataFrame AFDataFrame::average(std::string const &col, str_list group_by) const {
    AFDataFrame output;
    auto const &agg = _columns[_nameToCol.at(col)];
    if (agg.type() == STRING || agg.type() == DATE || agg.type() == TIME || agg.type() == DATETIME) {
        throw std::runtime_error("Expected numeric type to aggregate");
    }
    if (!group_by.size()) {
        output.add(Column(af::sum(agg.data(), 1) / agg.length()), "AVG("+col+")");
        return output;
    }
    af::array group_key = _columns[_nameToCol.at(*group_by.begin())].hash();
    int j = 0;
    for (auto i = group_by.begin() + 1; i != group_by.end(); ++i, j = !j) {
        if (!j) group_key = group_key ^ (_columns[_nameToCol.at(*i)].hash() << 2);
        else group_key = group_key ^ (_columns[_nameToCol.at(*i)].hash() >> 2);
    }
    auto to_sum(agg.data());
    af::array output_group_idx;
    sort(group_key, output_group_idx, group_key, 1);
    to_sum = to_sum(output_group_idx);

    auto diffe = diff1(group_key, 1) > 0;
    auto indexer = where64(join(1, af::constant(1,1,diffe.type()), diffe));
    auto key_count = hflat(where64(join(1, diffe, af::constant(1, 1, diffe.type()))) - indexer + 1); // histogram

    auto loops = af::sum<unsigned long long>(af::max(key_count, 1));
    auto summation = af::constant(0, key_count.dims(), to_sum.type());
    for (ull i = 0; i < loops; ++i) {
        auto b = i < key_count;
        summation(b) += (to_sum(indexer(b) + i) + 0);
    }

    output.add(Column(summation / key_count), "AVG("+col+")");
    for (const auto & i : group_by) {
        output.add(_columns[_nameToCol.at(i)].select(output_group_idx), i);
    }

    return output;
}

AFDataFrame AFDataFrame::count(std::string const &col, str_list group_by) const {
    AFDataFrame output;
    auto const &agg = _columns[_nameToCol.at(col)];
    if (!group_by.size()) {
        output.add(Column(af::array(agg.length(), u64)), "COUNT("+col+")");
        return output;
    }
    if (agg.type() == STRING || agg.type() == DATE || agg.type() == TIME || agg.type() == DATETIME) {
        throw std::runtime_error("Expected numeric type to aggregate");
    }
    af::array group_key = _columns[_nameToCol.at(*group_by.begin())].hash();
    int j = 0;
    for (auto i = group_by.begin() + 1; i != group_by.end(); ++i, j = !j) {
        if (!j) group_key = group_key ^ (_columns[_nameToCol.at(*i)].hash() << 2);
        else group_key = group_key ^ (_columns[_nameToCol.at(*i)].hash() >> 2);
    }

    af::array output_group_idx;
    sort(group_key, output_group_idx, group_key, 1);

    auto diffe = diff1(group_key, 1) > 0;
    auto indexer = where64(join(1, af::constant(1,1,diffe.type()), diffe));
    // ArrayFire's histogram may have a small bug that occurs when the integer sets are too big
    auto key_count = hflat(where64(join(1, diffe, af::constant(1,1,diffe.type()))) - indexer + 1); // histogram

    output.add(Column(key_count), "COUNT("+col+")");
    for (const auto & i : group_by) {
        output.add(_columns[_nameToCol.at(i)].select(output_group_idx), i);
    }

    return output;
}

AFDataFrame AFDataFrame::unionize(AFDataFrame &frame) const {
    if (_columns.size() != frame._columns.size()) throw std::runtime_error("Number of attributes do not match");
    auto out(*this);
    for (size_t i = 0; i < out._columns.size(); ++i)
        out._columns[i] = out._columns[i].concatenate(frame._columns[i]);
    return out;
}

void AFDataFrame::sortBy(unsigned int const col, bool const isAscending) {
    array key = _columns[col].hash(true);
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
        for (auto &i : _columns) i = i.select(main);
    } else {
        for (auto &i : _columns) i = i.select(idx);
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
    auto &left = _columns[lhs_column];
    auto &right = rhs._columns[rhs_column];
    if (left.type() != right.type()) throw std::runtime_error("Column type mismatch");
    if (left.isempty() || right.isempty()) return AFDataFrame();

    auto idx = hashCompare(left.hash(), right.hash());

    if (idx.first.isempty()) return AFDataFrame();

    if (left.type() == STRING) {
        af::array l = left.index(af::span, idx.first);
        af::array r = right.index(af::span, idx.second);
        auto keep = stringComp(left.data(), right.data(), l, r);
        idx.first = idx.first(keep);
        idx.second = idx.second(keep);
    }

    AFDataFrame result;
    for (size_t i = 0; i < columns(); i++) {
        result.add(_columns[i].select(idx.first), _colToName.at(i));
    }

    for (size_t i = 0; i < rhs.columns(); i++) {
        if (i == rhs_column) continue;
        result.add(rhs._columns[i].select(idx.second),
                   (rhs.name() + "." + rhs._colToName.at(i)));
    }

    return result;
}

std::pair<af::array, af::array> AFDataFrame::hashCompare(Column const &lhs, Column const &rhs) {
    if (lhs.type() != rhs.type()) throw std::runtime_error("Column type mismatch");
    if (lhs.isempty() || rhs.isempty()) return { af::array(0, u64), af::array(0, u64) };
    if (lhs.type() == STRING || lhs.type() == TIME || lhs.type() == DATE || lhs.type() == DATETIME) {
        return hashCompare(lhs.hash(), rhs.hash());
    }
    return hashCompare(lhs.data(), rhs.data());
}

std::pair<af::array, af::array> AFDataFrame::crossCompare(Column const &lhs, Column const &rhs) {
    if (lhs.type() != rhs.type()) throw std::runtime_error("Column type mismatch");
    if (lhs.isempty() || rhs.isempty()) return { af::array(0, u64), af::array(0, u64) };
    if (lhs.type() == STRING || lhs.type() == TIME || lhs.type() == DATE || lhs.type() == DATETIME) {
        return crossCompare(lhs.hash(), rhs.hash());
    }
    return crossCompare(lhs.data(), rhs.data());
}

std::pair<af::array, af::array> AFDataFrame::hashCompare(const array &left, const array &right) {
    if (left.isempty() || right.isempty()) return { af::array(0, u64), af::array(0, u64) };
    array lhs;
    array rhs;
    array idx;

    sort(lhs, idx, left, 1);
    lhs = join(0, lhs, idx.as(lhs.type()));
    sort(rhs, idx, right, 1);
    rhs = join(0, rhs, idx.as(rhs.type()));

    auto set = setUnique(rhs.row(0), true);
    auto set_num = af::sum<unsigned int>(diff1(lhs.row(0), 1) > 0) + 1;
    AFHashTable ht(std::move(set));
    if (set_num != ht.elements()) lhs = hashIntersect(lhs, ht);

    set = setUnique(lhs.row(0), true);
    set_num = af::sum<unsigned long long>(diff1(rhs.row(0), 1) > 0) + 1;
    ht = AFHashTable(std::move(set));
    if (set_num != ht.elements()) rhs = hashIntersect(rhs, ht);

    set_num = af::sum<unsigned long long>(diff1(rhs.row(0), 1) > 0) + 1;

    joinScatter(lhs, rhs, set_num);

    return { lhs, rhs };
}

std::pair<af::array, af::array> AFDataFrame::crossCompare(const array &left, const array &right) {
    if (left.isempty() || right.isempty()) return { af::array(0, u64), af::array(0, u64) };
    array lhs;
    array rhs;
    array idx;
    sort(lhs, idx, left, 1);
    lhs = join(0, lhs, idx.as(lhs.type()));
    sort(rhs, idx, right, 1);
    rhs = join(0, rhs, idx.as(rhs.type()));
    auto r_set = setUnique(rhs.row(0), true);
    lhs = crossIntersect(lhs, r_set);

    auto l_set = setUnique(lhs.row(0), true);
    rhs = crossIntersect(rhs, l_set);
    l_set = af::array();
    r_set = af::array();
    auto i_num = af::sum<unsigned int>(diff1(lhs.row(0), 1) > 0) + 1;

    joinScatter(lhs, rhs, i_num);

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
    if (_columns.empty()) return;
    for (auto &a : _columns) a.toHost();
}

void AFDataFrame::clear() {
    _columns.clear();
    _name.clear();
    _colToName.clear();
    _nameToCol.clear();
}




