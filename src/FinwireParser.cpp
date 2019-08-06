#include "include/FinwireParser.h"
#include "include/BatchFunctions.h"
#include "include/TPCDI_Utils.h"
#include <fstream>
using namespace af;
using namespace TPCDI_Utils;

FinwireParser::FinwireParser(std::vector<std::string> const &files) {
    auto text = collect(files);
    if (text.back() != '\n') text += '\n';
    _data = array(text.size() + 1, text.c_str()).as(u8);
    _data.eval();
    auto row_end = where64(_data == '\n');
    row_end = moddims(row_end, dim4(1, row_end.elements()));
    auto row_start = join(1, constant(0, 1, row_end.type()), row_end.cols(0, end - 1) + 1);
    _indexer = join(0, row_start, row_end);
    max(diff1(_indexer, 0)).scalar<ull>();
}

template<> Column FinwireParser::parse<char*>(array &start, unsigned int const length) const {
    auto idx = join(0, start, constant(length + 1, start.dims(), start.type()));
    auto out = stringGather(_data, idx);
    start += length;
    out(accum(idx.row(1), 1) - 1) = 0; // finwire files have no delimiter, need to add manually
    af::eval(start, idx, out);
    return Column(std::move(out), std::move(idx));
}
template<typename T>
Column FinwireParser::parse(array &start, unsigned int const length) const {
    Column output = parse<char*>(start, length);
    output.cast<T>();
    return output;
}
template Column FinwireParser::parse<unsigned char>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<short>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<unsigned short>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<int>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<unsigned int>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<long long>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<double>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<float>(array &start, unsigned int const length) const;
template Column FinwireParser::parse<unsigned long long>(array &start, unsigned int const length) const;

af::array FinwireParser::filterRowsByCategory(const FinwireParser::RecordType &type) const {
    using namespace BatchFunctions;
    auto recType = af::batchFunc(_indexer.row(0), range(dim4(3), 0, u64) + 15, batchAdd);
    af::array tmp = _data(recType);
    tmp = moddims(tmp, dim4(3, tmp.elements() / 3));
    tmp.eval();
    return allTrue(batchFunc(tmp, array(3, _search[type]), batchEqual), 0);
}

AFDataFrame FinwireParser::extractCmp() const {
    auto rows = filterRowsByCategory(CMP);
    af::array start = _indexer(0, rows);
    AFDataFrame output;
    ull const *lengths = _CMPLengths;

    for (int i = 0; i < _widths[CMP]; ++i) {
        if (i == 3) output.add(parse<unsigned long long>(start, *lengths));
        else output.add(parse<char*>(start, *lengths));
    }

    output.column(7).toDate(false, YYYYMMDD);
    output.column(0).toDateTime(false, YYYYMMDD);
    return output;
}

AFDataFrame FinwireParser::extractFin() const {
    auto rows = filterRowsByCategory(FIN);
    af::array start = _indexer(0, rows);
    AFDataFrame output;
    ull const *lengths = _FINLengths;
    output.add(parse<char*>(start, *lengths));
    output.add(parse<char*>(start, *lengths));
    output.add(parse<unsigned short>(start, *lengths));
    output.add(parse<unsigned char>(start, *lengths));
    output.add(parse<char*>(start, *lengths));
    output.add(parse<char*>(start, *lengths));
    for (int i = 0; i < 8; ++i) output.add(parse<double>(start, *lengths));
    output.add(parse<unsigned long long>(start, *lengths));
    output.add(parse<unsigned long long>(start, *lengths));

    output.column(4).toDate(false, YYYYMMDD);
    output.column(5).toDate(false, YYYYMMDD);
    output.column(0).toDateTime(false, YYYYMMDD);
    return output;
}

AFDataFrame FinwireParser::extractSec() const {
    auto rows = filterRowsByCategory(SEC);
    af::array start = _indexer(0, rows);
    AFDataFrame output;
    ull const *lengths = _SECLengths;
    for (int i = 0; i < 7; ++i) output.add(parse<double>(start, *lengths));
    output.add(parse<unsigned long long>(start, *lengths));
    output.add(parse<char*>(start, *lengths));
    output.add(parse<char*>(start, *lengths));
    output.add(parse<double>(start, *lengths));

    output.column(8).toDate(false, YYYYMMDD);
    output.column(9).toDate(false, YYYYMMDD);
    output.column(0).toDateTime(false, YYYYMMDD);
    return output;
}
