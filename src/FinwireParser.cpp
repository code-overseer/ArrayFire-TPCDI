#include "FinwireParser.h"
#include "BatchFunctions.h"
#include "Utils.h"
#include "KernelInterface.h"
#include "Logger.h"

typedef unsigned long long ull;
using namespace af;
using namespace Utils;

FinwireParser::FinwireParser(std::vector<std::string> const &files) {
    Logger::startTask("Finwire files concat");
    auto text = collect(files);
    if (text.back() != '\n') text += '\n';

    Logger::endLastTask();
    Logger::startTask("Finwire load");
    _data = array(text.size(), text.c_str()).as(u8);

    auto row_end = hflat(where64(_data == '\n'));
    auto row_start = join(1, constant(0, 1, row_end.type()), row_end.cols(0, end - 1) + 1);
    _indexer = join(0, row_start, row_end);
    _data.eval();
    _indexer.eval();
    Logger::endLastTask();
}

af::array FinwireParser::filterRowsByCategory(const FinwireParser::RecordType &type) const {
    using namespace BatchFunctions;
    auto recType = af::batchFunc(_indexer.row(0), range(dim4(3), 0, u64) + 15, batchAdd);
    af::array tmp = _data(recType);
    tmp = moddims(tmp, dim4(3, tmp.elements() / 3));
    return allTrue(batchFunc(tmp, array(3, _search[type]), batchEqual), 0);
}

AFDataFrame FinwireParser::extractCmp() const {
    callGC();
    AFDataFrame output;
    int const *lengths = _CMPLengths;
    print("CMP");
    Logger::startTask("CMP columns extraction");
    auto rows = filterRowsByCategory(CMP);
    af::array start = _indexer(0, rows);
    for (int i = 0; *lengths; ++lengths, ++i) {
        if (i == 3) output.add(parse<unsigned long long>(start, *lengths));
        else output.add(_extract(start, *lengths, CMP));

        if (i == 0) output(i).toDateTime(YYYYMMDD);
        else if (i == 7) output(i).toDate(false, YYYYMMDD);
        start += *lengths;
    }
    Logger::endLastTask();
    return output;
}

AFDataFrame FinwireParser::extractFin() const {
    callGC();
    AFDataFrame output;
    int const *lengths = _FINLengths;
    print("FIN");
    Logger::startTask("FIN columns extraction");

    auto rows = filterRowsByCategory(FIN);
    af::array start = _indexer(0, rows);
    for (int i = 0; *lengths; ++lengths, ++i) {
        if (i == 2) output.add(parse<unsigned short>(start, *lengths));
        else if (i == 3) output.add(parse<unsigned char>(start, *lengths));
        else if (i >= 6 && i < 14) output.add(parse<double>(start, *lengths));
        else if (i >= 14 && i < 16) output.add(parse<unsigned long long>(start, *lengths));
        else output.add(_extract(start, *lengths, FIN));

        if (i == 0) output(i).toDateTime(YYYYMMDD);
        else if (i == 4 || i == 5) output(i).toDate(false, YYYYMMDD);

        start += *lengths;
    }
    Logger::endLastTask();
    return output;
}

AFDataFrame FinwireParser::extractSec() const {
    callGC();
    print("SEC");
    int const *lengths = _SECLengths;
    AFDataFrame output;

    Logger::startTask("SEC columns extraction");
    auto rows = filterRowsByCategory(SEC);
    af::array start = _indexer(0, rows);

    for (int i = 0; *lengths; ++lengths, ++i) {
        if (i == 7) output.add(parse<unsigned long long>(start, *lengths));
        else if (i == 10) output.add(parse<double>(start, *lengths));
        else output.add(_extract(start, *lengths, SEC));

        if (i == 0) output(i).toDateTime(YYYYMMDD);
        else if (i == 8 || i == 9) output(i).toDate(false, YYYYMMDD);
        start += *lengths;
    }
    Logger::endLastTask();

    return output;
}

template<typename T>
Column FinwireParser::parse(const af::array& start, unsigned int const length) const {
    auto idx = join(0,start, constant(length + 1, start.dims(), start.type()));
    return Column(numericParse<T>(_data, idx));
}
template Column FinwireParser::parse<unsigned char>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<short>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<unsigned short>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<int>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<unsigned int>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<long long>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<double>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<float>(const af::array& start, unsigned int const length) const;
template Column FinwireParser::parse<unsigned long long>(const af::array& start, unsigned int const length) const;
template<> Column FinwireParser::parse<char*>(const af::array& start, unsigned int const length) const {
    if (!length) return Column(array(0, u8), array(0,u64));
    auto idx = join(0,start, constant(length + 1, start.dims(), start.type()));
    auto out = stringGather(_data, idx);
    return Column(std::move(out), std::move(idx));
}
Column FinwireParser::_extract(const array &start, unsigned int const length, FinwireParser::RecordType const &type) const {
    if (type != CMP && length == 60) {
        auto idx = join(0, start, hflat((_data(start) != '0') * 50 + 11).as(start.type()));
        auto out = stringGather(_data, idx);
        return Column(std::move(out), std::move(idx));
    }
    return parse<char*>(start, length);
}