#include <cstdio>
#include <cstdlib>
#include <rapidxml.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "BatchFunctions.h"
#include "Tests.h"
#include "TPC_DI.h"
#include "XMLFlattener.h"

using namespace af;
using namespace rapidxml;

namespace DIR {
  char const* HR = "/Users/bryanwong/Downloads/TPCData/HR3.csv";
  char const* DATE = "/Users/bryanwong/Downloads/TPCData/TestDate.csv";
  char const* UINT = "/Users/bryanwong/Downloads/TPCData/TestUint.csv";
  char const* UCHAR = "/Users/bryanwong/Downloads/TPCData/TestUchar.csv";
  char const* INT = "/Users/bryanwong/Downloads/TPCData/TestInt.csv";
  char const* FLOAT = "/Users/bryanwong/Downloads/TPCData/TestFloat.csv";
  char const* DIRECTORY = "/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/";
}


int main(int argc, char *argv[])
{
  setBackend(AF_BACKEND_OPENCL);
  setDevice(4);
  auto path = argc > 1 ? argv[1] : DIR::DIRECTORY;

  timer::start();
  auto batchDate = loadBatchDate(path);
  auto dimDate = loadDimDate(path);
  auto dimTime = loadDimTime(path);
 
  dimTime.flushToHost();
  print("industry");
  auto industry = loadIndustry(path);
  print("statusType");
  auto statusType = loadStatusType(path);
//  print("taxRate");
//  auto taxRate = loadTaxRate(path);
//  taxRate.flushToHost();
//  print("tradeType");
//  auto tradeType = loadTradeType(path);
//  tradeType.flushToHost();
//  print("audit");
//  auto audit = loadAudit(path);
//  audit.flushToHost();
  print("finwire");
  auto finwire = loadStagingFinwire(path);
//  print("s_prospect");
//  auto s_prospect = loadStagingProspect(path);
//  print("s_cash");
//  auto s_cash = loadStagingCashBalances(path);
//  print("s_watches");
//  auto s_watches = loadStagingWatches(path);
//  print("s_customer");
//  auto s_customer = loadStagingCustomer(path);
  print("dimCompany");
  auto dimCompany = loadDimCompany(*finwire.company, industry, statusType, dimDate);
  industry.flushToHost();
//  print("financial");
//  auto financial = loadFinancial(*finwire.financial, dimCompany);
//  financial.flushToHost();
//  print("dimSecurity");
//  auto dimSecurity = loadDimSecurity(*finwire.security, dimCompany, statusType);
  
//  dimSecurity.flushToHost();
  dimCompany.flushToHost();
  statusType.flushToHost();
  finwire.security->clear();
  finwire.company->clear();
  finwire.financial->clear();
//  print("prospect");
//  auto prospect = loadProspect(s_prospect, batchDate);
//  prospect.flushToHost();
//  batchDate.flushToHost();
//  s_prospect.clear();
//  print("dimBroker");
//  auto dimBroker = loadDimBroker(path, dimDate);
//  dimBroker.flushToHost();
//  dimDate.flushToHost();
  af::sync();
  char t[64];
  sprintf(t, "%f", timer::stop());
  
  if (argc < 3) return 0;
  std::ofstream outfile;
  outfile.open("results.csv", std::ios_base::app);
  outfile << argv[2] << ',' << t << '\n';
  outfile.close();
  
  return 0;
}
