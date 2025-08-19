/*
 * AutoGrid Embedded Controller (C++/ARM)
 * Real-time hardware interface and control
 * Target: ARM Cortex-M4/ESP32/STM32
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <ctime>

class SensorData {
public:
    float voltage;
    float current;
    float temperature;
    float frequency;
    std::chrono::steady_clock::time_point timestamp;
    
    SensorData() : voltage(0), current(0), temperature(0), frequency(0) {
        timestamp = std::chrono::steady_clock::now();
    }
};

class ActuatorControl {
public:
    bool relay_states[8];
    float pwm_outputs[4];
    bool emergency_stop;
    
    ActuatorControl() : emergency_stop(false) {
        for(int i = 0; i < 8; i++) relay_states[i] = false;
        for(int i = 0; i < 4; i++) pwm_outputs[i] = 0.0f;
    }
};

class EmbeddedController {
private:
    std::atomic<bool> running;
    SensorData sensors;
    ActuatorControl actuators;
    float system_frequency = 60.0f; // Hz
    
    // Hardware simulation (in real implementation, these would be GPIO/ADC operations)
    void readSensors() {
        // Simulate realistic power grid measurements
        auto now = std::chrono::steady_clock::now();
        auto time_since_epoch = now.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count();
        
        // Grid voltage (120V nominal with small variations)
        sensors.voltage = 120.0f + 2.0f * sin(millis * 0.001f) + 
                         ((rand() % 100) - 50) * 0.01f;
        
        // Current based on power demand
        float power_demand = 15.0f + 10.0f * sin(millis * 0.0005f) +
                           ((rand() % 100) - 50) * 0.1f;
        sensors.current = power_demand / sensors.voltage;
        
        // Temperature (ambient + load heating)
        sensors.temperature = 25.0f + power_demand * 0.5f + 
                            ((rand() % 100) - 50) * 0.05f;
        
        // Frequency stability
        sensors.frequency = system_frequency + ((rand() % 100) - 50) * 0.001f;
        
        sensors.timestamp = now;
    }
    
    void updateActuators() {
        // Voltage regulation
        if (sensors.voltage > 125.0f) {
            actuators.relay_states[0] = true; // Voltage regulator tap down
        } else if (sensors.voltage < 115.0f) {
            actuators.relay_states[1] = true; // Voltage regulator tap up
        } else {
            actuators.relay_states[0] = false;
            actuators.relay_states[1] = false;
        }
        
        // Overcurrent protection
        if (sensors.current > 50.0f) {
            actuators.emergency_stop = true;
            for(int i = 2; i < 6; i++) {
                actuators.relay_states[i] = false; // Trip breakers
            }
        }
        
        // Temperature-based fan control
        if (sensors.temperature > 45.0f) {
            actuators.pwm_outputs[0] = (sensors.temperature - 25.0f) / 50.0f; // Cooling fan
        } else {
            actuators.pwm_outputs[0] = 0.0f;
        }
        
        // Frequency regulation
        if (sensors.frequency < 59.9f) {
            actuators.pwm_outputs[1] = 0.8f; // Increase generation
        } else if (sensors.frequency > 60.1f) {
            actuators.pwm_outputs[1] = 0.2f; // Decrease generation
        } else {
            actuators.pwm_outputs[1] = 0.5f; // Nominal
        }
    }
    
    void emergencyProtection() {
        // Critical safety checks (must execute within 1ms)
        if (sensors.voltage > 130.0f || sensors.voltage < 100.0f) {
            actuators.emergency_stop = true;
            std::cout << "[EMERGENCY] Voltage out of range: " << sensors.voltage << "V" << std::endl;
        }
        
        if (sensors.current > 100.0f) {
            actuators.emergency_stop = true;
            std::cout << "[EMERGENCY] Overcurrent detected: " << sensors.current << "A" << std::endl;
        }
        
        if (sensors.temperature > 80.0f) {
            actuators.emergency_stop = true;
            std::cout << "[EMERGENCY] Overtemperature: " << sensors.temperature << "°C" << std::endl;
        }
        
        if (abs(sensors.frequency - system_frequency) > 2.0f) {
            actuators.emergency_stop = true;
            std::cout << "[EMERGENCY] Frequency deviation: " << sensors.frequency << "Hz" << std::endl;
        }
    }
    
public:
    EmbeddedController() : running(false) {
        srand(time(nullptr));
    }
    
    void start() {
        running = true;
        std::cout << "[C++] Embedded Controller starting..." << std::endl;
        std::cout << "[C++] Initializing GPIO pins..." << std::endl;
        std::cout << "[C++] Configuring ADC channels..." << std::endl;
        std::cout << "[C++] Setting up interrupt handlers..." << std::endl;
        std::cout << "[C++] Controller ready for real-time operation" << std::endl;
    }
    
    void stop() {
        running = false;
        actuators.emergency_stop = true;
        std::cout << "[C++] Embedded Controller stopping..." << std::endl;
        std::cout << "[C++] Safe shutdown completed" << std::endl;
    }
    
    void controlLoop() {
        while (running) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Real-time control loop (target: 1kHz = 1ms cycle time)
            readSensors();
            emergencyProtection();
            updateActuators();
            
            // Ensure deterministic timing
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            if (duration.count() > 1000) { // More than 1ms
                std::cout << "[WARNING] Control loop overrun: " << duration.count() << "μs" << std::endl;
            }
            
            // Sleep for remainder of 1ms cycle
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    SensorData getSensorData() const {
        return sensors;
    }
    
    ActuatorControl getActuatorStates() const {
        return actuators;
    }
    
    void setSafeMode() {
        // Implement safe operating parameters
        for(int i = 0; i < 8; i++) {
            actuators.relay_states[i] = false;
        }
        for(int i = 0; i < 4; i++) {
            actuators.pwm_outputs[i] = 0.0f;
        }
        actuators.emergency_stop = false;
        std::cout << "[C++] Safe mode activated" << std::endl;
    }
    
    void resetEmergencyStop() {
        if (sensors.voltage > 110.0f && sensors.voltage < 130.0f &&
            sensors.current < 80.0f && sensors.temperature < 70.0f) {
            actuators.emergency_stop = false;
            std::cout << "[C++] Emergency stop reset - normal operation resumed" << std::endl;
        } else {
            std::cout << "[C++] Cannot reset emergency stop - unsafe conditions" << std::endl;
        }
    }
    
    void printStatus() const {
        std::cout << "\n=== EMBEDDED CONTROLLER STATUS ===" << std::endl;
        std::cout << "Voltage: " << sensors.voltage << "V" << std::endl;
        std::cout << "Current: " << sensors.current << "A" << std::endl;
        std::cout << "Temperature: " << sensors.temperature << "°C" << std::endl;
        std::cout << "Frequency: " << sensors.frequency << "Hz" << std::endl;
        std::cout << "Emergency Stop: " << (actuators.emergency_stop ? "ACTIVE" : "CLEAR") << std::endl;
        
        std::cout << "Relay States: ";
        for(int i = 0; i < 8; i++) {
            std::cout << (actuators.relay_states[i] ? "1" : "0");
        }
        std::cout << std::endl;
        
        std::cout << "PWM Outputs: ";
        for(int i = 0; i < 4; i++) {
            std::cout << actuators.pwm_outputs[i] << " ";
        }
        std::cout << std::endl;
    }
};

// Python interface functions (extern "C" for FFI)
extern "C" {
    EmbeddedController* controller_instance = nullptr;
    
    void init_embedded_controller() {
        if (!controller_instance) {
            controller_instance = new EmbeddedController();
            controller_instance->start();
        }
    }
    
    void start_control_loop() {
        if (controller_instance) {
            std::thread control_thread(&EmbeddedController::controlLoop, controller_instance);
            control_thread.detach();
        }
    }
    
    void stop_embedded_controller() {
        if (controller_instance) {
            controller_instance->stop();
            delete controller_instance;
            controller_instance = nullptr;
        }
    }
    
    void get_sensor_data(float* voltage, float* current, float* temperature, float* frequency) {
        if (controller_instance) {
            SensorData data = controller_instance->getSensorData();
            *voltage = data.voltage;
            *current = data.current;
            *temperature = data.temperature;
            *frequency = data.frequency;
        }
    }
    
    void emergency_stop() {
        if (controller_instance) {
            controller_instance->setSafeMode();
        }
    }
    
    void reset_emergency() {
        if (controller_instance) {
            controller_instance->resetEmergencyStop();
        }
    }
}

// Standalone test program
int main() {
    std::cout << "AutoGrid Embedded Controller Test" << std::endl;
    std::cout << "===================================" << std::endl;
    
    EmbeddedController controller;
    controller.start();
    
    // Run for 10 seconds
    std::thread control_thread(&EmbeddedController::controlLoop, &controller);
    
    for (int i = 0; i < 10; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        controller.printStatus();
    }
    
    controller.stop();
    control_thread.join();
    
    return 0;
}