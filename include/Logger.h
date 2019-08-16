#ifndef ARRAYFIRE_TPCDI_LOGGER_H
#define ARRAYFIRE_TPCDI_LOGGER_H
#include <unordered_map>
#include <arrayfire.h>
#include <iostream>
#include <fstream>
#include <vector>
#ifdef ITT_ENABLED
    #include <ittnotify.h>
#endif

namespace Logger {
    static std::unordered_map<std::string, std::vector<double>> _times;
    static std::unordered_map<std::string, af::timer> _timers;
    static std::string directory;

    void startTimer(std::string const &name = "main");
    void logTime(std::string const &name = "main", bool show = true);
    void sendToCSV();

    #ifdef ITT_ENABLED
    static __itt_domain* _domain = __itt_domain_create("AF_tpcdi");
    #endif
    inline void startCollection() {
        #ifdef ITT_ENABLED
        __itt_resume();
        #endif
    }
    inline void pauseCollection() {
        #ifdef ITT_ENABLED
        __itt_pause();
        #endif
    }
    inline void startTask(char const* taskname) {
        #ifdef ITT_ENABLED
        auto task = __itt_string_handle_create(taskname);
        __itt_task_begin(_domain, __itt_null, __itt_null, task);
        #endif
    }
    inline void endLastTask() {
        #ifdef ITT_ENABLED
        __itt_task_end(_domain);
        #endif
    }
};

#endif //ARRAYFIRE_TPCDI_LOGGER_H
