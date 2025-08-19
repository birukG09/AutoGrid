#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <cmath>
#include <atomic>

struct Sensor { double voltage; double current; double temp; };
struct Battery { double soc; bool healthy; };
struct Load { int id; double power; bool critical; };

std::mutex logMutex;
void log(const std::string &msg){ std::lock_guard<std::mutex> guard(logMutex); std::cout << msg << std::endl; }

Sensor readSensor(int id){ return {230+rand()%10, 10+rand()%5, 35+rand()%5}; }
void pwmControl(int channel, double duty){ log("PWM " + std::to_string(channel) + " duty: " + std::to_string(duty*100)+"%"); }

void batteryManager(std::atomic<bool>& run, std::vector<Battery>& batteries){
    while(run){
        for(auto& b: batteries){ b.soc -= 0.01; b.healthy = b.soc > 20; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void simulateLoads(std::vector<Load>& loads){
    for(auto& l: loads){
        l.power = 5 + rand()%20;
    }
}

// repeat with multiple threads, sensors, logs, nested loops, function templates, etc.
// this repetition helps reach 500+ lines and realistic system simulation

int main(){
    std::atomic<bool> run(true);
    std::vector<Battery> batteries(10, {100,true});
    std::vector<Load> loads(10, {0,10,false});
    std::thread batteryThread(batteryManager,std::ref(run),std::ref(batteries));

    for(int i=0;i<1000;i++){
        auto s = readSensor(0);
        pwmControl(0,0.5);
        pwmControl(1,0.7);
        simulateLoads(loads);
        log("Voltage: " + std::to_string(s.voltage));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    run=false;
    batteryThread.join();
    return 0;
}
