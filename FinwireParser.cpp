#include "FinwireParser.h"
#include "BatchFunctions.h"
#include "TPCDI_Utils.h"

using namespace af;
using namespace BatchFunctions;
using namespace TPCDI_Utils;

FinwireParser::FinwireParser(std::vector<std::string> const &files) {
    auto text = collect(files);
    if (text.back() != '\n') text += '\n';
    _finwireData = array(text.size() + 1, text.c_str()).as(u8);
    _finwireData = _finwireData(where(_finwireData != '\r'));
    _finwireData.eval();

    auto row_end = where(_finwireData == '\n');
    row_end = moddims(row_end, dim4(1, row_end.elements()));
    auto row_start = join(1, constant(0, 1, u32), row_end.cols(0, end - 1) + 1);
    _indexer = join(0, row_start, row_end);
    _maxRowWidth = max(diff1(_indexer, 0)).scalar<uint32_t>();
}

af::array FinwireParser::_extract(af::array const &start, int const length) const {
    array column;
    column = batchFunc(start.row(0), range(dim4(length, 1), 0, u32), batchAdd);
    column(where(batchFunc(column, start.row(1), batchGE))) = UINT32_MAX;
    {
        auto cond = where(column != UINT32_MAX);
        array tmp = column(cond);
        tmp = _finwireData(tmp).as(u32);
        tmp.eval();
        column(cond) = tmp;
    }
    column(where(column == UINT32_MAX)) = 0;
    column = join(0, column.as(u8), constant(0, dim4(1, column.dims(1)), u8));
    column.eval();
    return column;
}

AFDataFrame FinwireParser::extractData(RecordType const type) const {
    AFDataFrame output;
    array rows;
    {
        auto recType = batchFunc(_indexer.row(0), range(dim4(3, 1), 0, u32) + 15, batchAdd);
        array tmp = _finwireData(recType);

        tmp = moddims(tmp, dim4(3, tmp.dims(0) / 3));
        tmp.eval();

        rows = where(allTrue(batchFunc(tmp, array(3, 1, _search[type]), batchEqual), 0));
        if (rows.isempty()) {
            for (int i = 0; i < _widths[type]; ++i) output.add(array(1, 0, u8), STRING);
            output.data(0) = _PTSToDatetime(output.data(0), false, MMDDYYYY);
            output.types(0) = DATETIME;
            return output;
        }
    }
    int const *lengths;
    switch (type) {
        case CMP:
            lengths = _CMPLengths;
            break;
        case FIN:
            lengths = _FINLengths;
            break;
        case SEC:
            lengths = _SECLengths;
            break;
        default:
            throw std::runtime_error("Invalid type");
    }

    array start = _indexer(span, rows);
    for (; *lengths != -1; lengths++) {
        output.add(_extract(start, *lengths), STRING);
        start(0, span) += *lengths;
    }
    output.data(0) = _PTSToDatetime(output.data(0));
    output.types(0) = DATETIME;
    return output;
}

AFDataFrame FinwireParser::extractCmp() const {
    auto output = extractData(CMP);
    output.data(7) = stringToDate(output.data(7));
    output.types(7) = DATE;
    return output;
}

AFDataFrame FinwireParser::extractFin() const {
    auto output = extractData(FIN);

    output.data(2) = stringToNum(output.data(2), u16);
    output.types(2) = USHORT;
    output.data(3) = stringToNum(output.data(3), u8);
    output.types(3) = UCHAR;
    output.data(4) = stringToDate(output.data(4));
    output.types(4) = DATE;
    output.data(5) = stringToDate(output.data(5));
    output.types(5) = DATE;
    output.data(6) = stringToNum(output.data(6), f64);
    output.types(6) = DOUBLE;
    output.data(7) = stringToNum(output.data(7), f64);
    output.types(7) = DOUBLE;
    output.data(8) = stringToNum(output.data(8), f64);
    output.types(8) = DOUBLE;
    output.data(9) = stringToNum(output.data(9), f64);
    output.types(9) = DOUBLE;
    output.data(10) = stringToNum(output.data(10), f64);
    output.types(10) = DOUBLE;
    output.data(11) = stringToNum(output.data(11), f64);
    output.types(11) = DOUBLE;
    output.data(12) = stringToNum(output.data(12), f64);
    output.types(12) = DOUBLE;
    output.data(13) = stringToNum(output.data(13), f64);
    output.types(13) = DOUBLE;
    output.data(14) = stringToNum(output.data(14), u64);
    output.types(14) = U64;
    output.data(15) = stringToNum(output.data(15), u64);
    output.types(15) = U64;

    return output;
}

AFDataFrame FinwireParser::extractSec() const {
    auto output = extractData(SEC);

    output.data(7) = stringToNum(output.data(7), u64);
    output.types(7) = U64;
    output.data(8) = stringToDate(output.data(8));
    output.types(8) = DATE;
    output.data(9) = stringToDate(output.data(9));
    output.types(9) = DATE;
    output.data(10) = stringToNum(output.data(10), f64);
    output.types(10) = DOUBLE;
    return output;
}

af::array FinwireParser::_PTSToDatetime(af::array &PTS, bool isDelimited, DateFormat inputFormat) {
    if (!PTS.dims(1)) return array(6, 0, u16);

    af::array date = PTS.rows(0, isDelimited ? 10 : 8);
    date = stringToDate(date, isDelimited, inputFormat);

    af::array time = PTS.rows(end - (isDelimited ? 8 : 6), end);
    time = stringToTime(time, isDelimited);

    return join(0, date, time);
}