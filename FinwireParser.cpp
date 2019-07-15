//
// Created by Bryan Wong on 2019-07-03.
//

#include "FinwireParser.h"
#include "BatchFunctions.h"
using namespace af;
using namespace BatchFunctions;

FinwireParser::FinwireParser(char const *filename) {
    auto text = loadFile(filename);
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
            output.data(0) = _PTSToDatetime(output.data(0));
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

array FinwireParser::stringToDate(af::array &datestr, DateFormat inputFormat, bool isDelimited) {
    if (!datestr.dims(1)) return array(3, 0, u16);

    af::array out = datestr.rows(0, end - 1);
    auto nulls = where(allTrue(out == 0 || out == ' ', 0));
    nulls.eval();
    out(where(out >= '0' && out <='9')) -= '0';
    out(span, nulls) = 0;
    if (isDelimited) {
        auto delims = AFParser::dateDelimIndices(inputFormat);
        out(seq(delims.first, delims.second, delims.second - delims.first), span) = 255;
    }
    out = moddims(out(where(out >= 0 && out <= 9)), dim4(8, out.dims(1)));
    out = batchFunc(out, flip(pow(10, range(dim4(8, 1), 0, u32)), 0), batchMult);
    out = sum(out, 0);

    AFParser::dateKeyToDate(out, inputFormat);

    out.eval();

    return out;
}

array FinwireParser::stringToTime(af::array &timestr, bool isDelimited) {
    if (!timestr.dims(1)) return array(3, 0, u16);

    af::array out = timestr.rows(0, end - 1);
    auto nulls = where(allTrue(out == 0, 0));
    nulls.eval();
    out = out - '0';
    out(span, nulls) = 0;
    if (isDelimited) {
        out(seq(2, 5, 3), span) = 255;
    }
    out = moddims(out(where(out >= 0 && out <= 9)), dim4(6, out.dims(1)));
    out = batchFunc(out, flip(pow(10, range(dim4(6, 1), 0, u32)), 0), batchMult);
    out = sum(out, 0);
    out = join(0, out / 10000, out % 10000 / 100, out % 100).as(u16);
    out.eval();

    return out;
}

array FinwireParser::_PTSToDatetime(array &PTS, DateFormat inputFormat, bool isDelimited) {
    if (!PTS.dims(1)) return array(6, 0, u16);

    af::array date = PTS.rows(0, isDelimited ? 10 : 8);
    auto nulls = where(allTrue(date == 0, 0));
    nulls.eval();

    date -= '0';
    date(span, nulls) = 0;

    if (isDelimited) {
        auto delims = AFParser::dateDelimIndices(inputFormat);
        date(seq(delims.first, delims.second, delims.second - delims.first), span) = 255;
    }

    date = moddims(date(where(date >= 0 && date <= 9)), dim4(8, date.dims(1)));
    date = batchFunc(date, flip(pow(10, range(dim4(8, 1), 0, u32)), 0), batchMult);
    date = sum(date, 0);
    AFParser::dateKeyToDate(date, inputFormat);
    date.eval();

    af::array time = PTS.rows(end - (isDelimited ? 8 : 6), end - 1);

    time -= '0';
    time(span, nulls) = 0;
    if (isDelimited) {
        time(seq(2, 5, 3), span) = 255;
    }

    time = moddims(time(where(time >= 0 && time <= 9)), dim4(6, time.dims(1)));
    time = batchFunc(time, flip(pow(10, range(dim4(6, 1), 0, u32)), 0), batchMult);
    time = sum(time, 0);
    time = join(0, time / 10000, time % 10000 / 100, time % 100).as(u16);
    time.eval();

    return join(0, date, time);
}

af::array FinwireParser::stringToNum(af::array &numstr, af::dtype type) {

    if (!numstr.dims(1)) return array(0, type);

    auto lengths = numstr;
    lengths(where(lengths)) = 255;
    {
        auto tmp = where(lengths == 0);
        lengths(tmp) = (tmp % lengths.dims(0)).as(u8);
    }
    lengths = min(lengths, 0);

    af::array output = numstr.rows(0, end - 1);
    output(where(output == ' ')) = '0';
    auto const maximum = output.dims(0);
    if (!maximum) return constant(0, numstr.dims(1), type);

    array negatives;
    array points;

    auto cond = where(output == '-');
    negatives = cond / maximum;
    negatives = moddims(negatives, dim4(1, negatives.dims(0)));
    output(cond) = 255;

    cond = where(output == '.');
    points = constant(-1, 1, output.dims(0) / maximum, s32);
    points(cond / maximum) = (cond % maximum).as(s32);
    output(cond) = 255;

    output(where(output >= '0' && output <= '9')) -= '0';
    output = output.as(type);
    output.eval();
    {
        auto exponent = batchFunc(flip(range(dim4(maximum, 1), 0, s32),0) - maximum, lengths, batchAdd);
        exponent = batchFunc(exponent, points, batchSub);
        exponent(where(exponent > 0)) -= 1;
        exponent = pow(10, exponent.as(type));
        output(where(output == 255)) = 0;
        output *= exponent;
    }
    output = sum(output, 0).as(type);
    if (!negatives.isempty()) output(negatives) *= -1;
    return output;
}