#ifndef ARRAYFIRE_TPCDI_LOGGER_H
#define ARRAYFIRE_TPCDI_LOGGER_H
#include <unordered_map>
#include <arrayfire.h>
#include <iostream>
#include <fstream>
#include <vector>

class Logger {
private:
    Logger() = default;
    std::vector<std::pair<std::string, double>> _times;
    std::unordered_map<std::string, af::timer> _timers;
public:
    static Logger& instance();
    static std::string& output();
    static void startTimer(std::string const &name = "main");
    static void logTime(std::string const &name = "main", bool show = true);
    static void sendToCSV();
    Logger(Logger const &other) = delete;
    Logger& operator=(Logger const &other) = delete;
    Logger(Logger &&other) = delete;
    Logger& operator=(Logger &&other) = delete;
};

#endif //ARRAYFIRE_TPCDI_LOGGER_H
