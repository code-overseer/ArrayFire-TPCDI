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

Column FinwireParser::_extract(array &start, unsigned int const length) const {
    auto idx = join(0, start, constant(length + 1, start.dims(), start.type()));
    auto out = stringGather(_data, idx);
    start += length;
    out(accum(idx.row(1), 1) - 1) = 0; // finwire files have no delimiter, need to add manually
    start.eval();
    out.eval();
    idx.eval();
    return Column(std::move(out), std::move(idx));
}

af::array FinwireParser::filterRowsByCategory(const FinwireParser::RecordType &type) const {
    using namespace BatchFunctions;
    auto recType = af::batchFunc(_indexer.row(0), range(dim4(3), 0, u64) + 15, batchAdd);
    af::array tmp = _data(recType);
    tmp = moddims(tmp, dim4(3, tmp.elements() / 3));
    return allTrue(batchFunc(tmp, array(3, _search[type]), batchEqual), 0);
}

AFDataFrame FinwireParser::extractCmp() const {
    auto rows = filterRowsByCategory(CMP);
    af::array start = _indexer(0, rows);
    AFDataFrame output;
    ull const *lengths = _CMPLengths;

    while(*lengths) {
        output.add(_extract(start, *lengths));
        ++lengths;
    }
    output.column(0).toDateTime(false, YYYYMMDD);
    output.column(3).cast<unsigned long long>();
    output.column(7).toDate(false, YYYYMMDD);
    return output;
}

AFDataFrame FinwireParser::extractFin() const {
    auto rows = filterRowsByCategory(FIN);
    af::array start = _indexer(0, rows);

    AFDataFrame output;
    ull const *lengths = _FINLengths;
    while(*lengths) {
        output.add(_extract(start, *lengths));
        ++lengths;
    }
    output.column(0).toDateTime(false, YYYYMMDD);
    output.column(2).cast<unsigned short>();
    output.column(3).cast<unsigned char>();
    output.column(4).toDate(false, YYYYMMDD);
    output.column(5).toDate(false, YYYYMMDD);
    for (int i = 6; i < 14; ++i) output.column(i).cast<double>();
    output.column(14).cast<unsigned long long>();
    output.column(15).cast<unsigned long long>();
    return output;
}

AFDataFrame FinwireParser::extractSec() const {
    auto rows = filterRowsByCategory(SEC);
    af::array start = _indexer(0, rows);
    AFDataFrame output;
    ull const *lengths = _SECLengths;
    while(*lengths) {
        output.add(_extract(start, *lengths));
        ++lengths;
    }

    output.column(0).toDateTime(false, YYYYMMDD);
    output.column(7).cast<unsigned long long>();
    output.column(8).toDate(false, YYYYMMDD);
    output.column(9).toDate(false, YYYYMMDD);
    output.column(10).cast<double>();
    return output;
}
