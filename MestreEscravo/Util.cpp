#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <cstdlib>

namespace Util{
    class ClockTime {
    private:
        static ClockTime* instancia;

        std::chrono::steady_clock::time_point time_begin;
        std::chrono::steady_clock::time_point time_end;
        int time_stamp;
        int time_buffer;

        ClockTime() {
            time(NULL);
            this->time_stamp = 0;
            this->time_buffer = 0;
        }

        void mark_time(std::chrono::steady_clock::time_point* time) {
            *time = std::chrono::steady_clock::now();
        }

        void mark_time_result() {
            this->time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(this->time_end - this->time_begin).count();
            this->time_buffer += this->time_stamp;
        }
    
    public:
        //https://www.geeksforgeeks.org/implementation-of-singleton-class-in-cpp/
        // deleting copy constructor
        ClockTime(const ClockTime& obj)
            = delete;

        static ClockTime* getInstancia() {
            if (instancia == NULL)
                instancia = new ClockTime();

            return instancia;
        }

        void reset_time_buffer() {
            this->time_buffer = 0;
        }


        int get_time_buffer() {
            this->time_buffer;
        }

        void mark_begin_clock() {
            this->mark_time(&time_begin);
        }

        void mark_end_clock() {
            this->mark_time(&time_end);
            this->mark_time_result();
        }

        void present_time_stamp(std::string arg) {
            printf("%s.started at: %dns \t.finished at: %dns \t.costs (time): %dns\n", arg, this->time_begin, this->time_end, this->time_stamp);
        }

        void present_time_buffer(std::string arg) {
            printf("%s.all costs (time): %dns\n", arg, this->time_buffer);
        }
    };

    //ClockTime* ClockTime::instancia = new ClockTime();
}
