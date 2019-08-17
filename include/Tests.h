//
// Created by Bryan Wong on 2019-06-27.
//

#ifndef ARRAYFIRE_TPCDI_TESTS_H
#define ARRAYFIRE_TPCDI_TESTS_H

#include "AFParser.h"
#include "AFDataFrame.h"

void test_SignedInt(char const *filepath);
void test_Float(char const *filepath);
void test_UnsignedInt(char const *filepath);
void test_UnsignedLong(char const *filepath);
void test_SignedLong(char const *filepath);
void test_Double(char const *filepath);
void test_String(char const *filepath);

void test_Date(char const *filepath);
void test_Time(char const *filepath);
void test_stringToBool(char const *filepath);
void test_UChar(char const *filepath);
void test_StringHash(char const *filepath);

void testSetJoin();

#endif //ARRAYFIRE_TPCDI_TESTS_H
