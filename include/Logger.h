#ifndef ARRAYFIRE_TPCDI_LOGGER_H
#define ARRAYFIRE_TPCDI_LOGGER_H

#include <unordered_map>
#include <arrayfire.h>
#include <iostream>
#include <fstream>
#include <vector>

#ifdef ENABLE_ITT
    #include <ittnotify.h>
#endif

namespace Logger {
    static std::unordered_map<std::string, std::vector<double>> _times;
    static std::unordered_map<std::string, af::timer> _timers;
    inline std::string& directory(std::string const &dir = "") {
        static std::string directory;
        if (!dir.empty()) directory = dir;
        return directory;
    }

    void startTimer(std::string const &name = "main");

    void logTime(std::string const &name = "main", bool show = true);

    void sendToCSV(int const scale);

    #ifdef ENABLE_ITT
    static __itt_domain* _domain = __itt_domain_create("AF_tpcdi");
    #endif

    inline void startCollection() {
        #ifdef ENABLE_ITT
        af::sync();
        __itt_resume();
        #endif
    }

    inline void pauseCollection() {
        #ifdef ENABLE_ITT
        af::sync();
        __itt_pause();
        #endif
    }

    inline void startTask(char const *taskname) {
        #ifdef ENABLE_ITT
        auto task = __itt_string_handle_create(taskname);
        af::sync();
        __itt_task_begin(_domain, __itt_null, __itt_null, task);
        #endif
    }

    inline void endLastTask() {
        #ifdef ENABLE_ITT
        af::sync();
        __itt_task_end(_domain);
        #endif
    }
    
};

#endif //ARRAYFIRE_TPCDI_LOGGER_H
