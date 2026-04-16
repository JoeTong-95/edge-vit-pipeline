// data-transmit.cpp
// Jesse McDonald, 3-25-26
// Input: .csv data from traffic camera
// Converts .csv data to binary bytes
// Sends data to TTN

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------- CONFIG ----------------
static const std::string config_serial_port   = "/dev/ttyUSB0";
static const speed_t     config_baud_rate     = B9600;
static const std::string config_csv_path      = "data.csv";
static const int         config_send_delay_ms = 15000;   // keep conservative for TTN duty/fair-use
// ----------------------------------------

// Adjust these for your specific modem if needed.
struct lorawan_modem_commands {
    std::string at_ping            = "AT\r\n";
    std::string at_join            = "AT+JOIN\r\n";
    std::string at_send_hex_prefix = "AT+MSGHEX=\"";
    std::string at_send_hex_suffix = "\"\r\n";
};

static const lorawan_modem_commands modem_cmds;

// ---------- serial ----------
int open_serial_port(const std::string& device, speed_t baud) {
    int fd = open(device.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        throw std::runtime_error("Failed to open serial port: " + device);
    }

    struct termios tty {};
    if (tcgetattr(fd, &tty) != 0) {
        close(fd);
        throw std::runtime_error("tcgetattr failed");
    }

    cfsetospeed(&tty, baud);
    cfsetispeed(&tty, baud);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 20;   // 2 seconds

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        close(fd);
        throw std::runtime_error("tcsetattr failed");
    }

    return fd;
}

void write_all(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = write(fd, data.data() + sent, data.size() - sent);
        if (n < 0) {
            throw std::runtime_error("Serial write failed");
        }
        sent += static_cast<size_t>(n);
    }
}

std::string read_serial_response(int fd) {
    std::string response;
    char buffer[256];

    while (true) {
        ssize_t n = read(fd, buffer, sizeof(buffer));
        if (n < 0) {
            throw std::runtime_error("Serial read failed");
        }
        if (n == 0) {
            break;
        }

        response.append(buffer, buffer + n);

        if (response.find("OK") != std::string::npos ||
            response.find("ERROR") != std::string::npos ||
            response.find("Done") != std::string::npos ||
            response.find("joined") != std::string::npos) {
            break;
        }
    }

    return response;
}

std::string send_at_command(int fd, const std::string& cmd, int wait_ms = 1000) {
    write_all(fd, cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    std::string response = read_serial_response(fd);

    std::cout << ">> " << cmd;
    std::cout << "<< " << response << "\n";

    return response;
}

// ---------- CSV ----------
std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        size_t start = item.find_first_not_of(" \t\r\n");
        size_t end   = item.find_last_not_of(" \t\r\n");

        if (start == std::string::npos) {
            fields.push_back("");
        } else {
            fields.push_back(item.substr(start, end - start + 1));
        }
    }

    return fields;
}

bool is_csv_header_row(const std::vector<std::string>& fields) {
    if (fields.empty()) return false;
    return fields[0].find_first_not_of("0123456789") != std::string::npos;
}

// ---------- payload packing ----------
void append_uint16_le(std::vector<uint8_t>& output, uint16_t value) {
    output.push_back(static_cast<uint8_t>( value       & 0xFF));
    output.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

std::vector<uint8_t> build_lorawan_payload_from_csv_row(const std::vector<std::string>& fields) {
    // CSV format:
    // timestamp,object_id,vehicle_class,truck_type,count
    if (fields.size() < 5) {
        throw std::runtime_error("Expected 5 CSV fields: timestamp,object_id,vehicle_class,truck_type,count");
    }

    // Skip timestamp for airtime efficiency
    unsigned long object_id     = std::stoul(fields[1]);
    unsigned long vehicle_class = std::stoul(fields[2]);
    unsigned long truck_type    = std::stoul(fields[3]);
    unsigned long count         = std::stoul(fields[4]);

    if (object_id > 0xFFFFUL)  throw std::runtime_error("object_id out of uint16 range");
    if (vehicle_class > 0xFFUL) throw std::runtime_error("vehicle_class out of uint8 range");
    if (truck_type > 0xFFUL)    throw std::runtime_error("truck_type out of uint8 range");
    if (count > 0xFFUL)         throw std::runtime_error("count out of uint8 range");

    std::vector<uint8_t> payload;
    payload.reserve(5);

    append_uint16_le(payload, static_cast<uint16_t>(object_id));
    payload.push_back(static_cast<uint8_t>(vehicle_class));
    payload.push_back(static_cast<uint8_t>(truck_type));
    payload.push_back(static_cast<uint8_t>(count));

    return payload;
}

std::string bytes_to_hex_string(const std::vector<uint8_t>& bytes) {
    static const char hex_digits[] = "0123456789ABCDEF";

    std::string hex;
    hex.reserve(bytes.size() * 2);

    for (uint8_t b : bytes) {
        hex.push_back(hex_digits[(b >> 4) & 0x0F]);
        hex.push_back(hex_digits[b & 0x0F]);
    }

    return hex;
}

// ---------- modem ----------
void verify_modem_alive(int fd) {
    std::string response = send_at_command(fd, modem_cmds.at_ping, 500);
    if (response.find("OK") == std::string::npos) {
        throw std::runtime_error("Modem did not respond to AT");
    }
}

void join_network(int fd) {
    std::string response = send_at_command(fd, modem_cmds.at_join, 10000);

    if (response.find("OK") == std::string::npos &&
        response.find("joined") == std::string::npos &&
        response.find("Done") == std::string::npos) {
        throw std::runtime_error("Join failed");
    }
}

bool send_lorawan_uplink_hex(int fd, const std::string& payload_hex) {
    std::string command =
        modem_cmds.at_send_hex_prefix +
        payload_hex +
        modem_cmds.at_send_hex_suffix;

    std::string response = send_at_command(fd, command, 5000);

    if (response.find("OK") != std::string::npos ||
        response.find("Done") != std::string::npos) {
        return true;
    }

    return false;
}

// ---------- main ----------
int main() {
    int serial_fd = -1;

    try {
        serial_fd = open_serial_port(config_serial_port, config_baud_rate);

        verify_modem_alive(serial_fd);
        join_network(serial_fd);

        std::ifstream csv_file(config_csv_path);
        if (!csv_file.is_open()) {
            throw std::runtime_error("Could not open CSV file: " + config_csv_path);
        }

        std::string line;
        bool first_row = true;

        while (std::getline(csv_file, line)) {
            if (line.empty()) {
                continue;
            }

            try {
                auto fields = split_csv_row(line);

                if (first_row) {
                    first_row = false;
                    if (is_csv_header_row(fields)) {
                        std::cout << "Skipping header row\n";
                        continue;
                    }
                }

                std::vector<uint8_t> payload = build_lorawan_payload_from_csv_row(fields);
                std::string payload_hex = bytes_to_hex_string(payload);

                std::cout << "Payload hex: " << payload_hex << "\n";

                if (!send_lorawan_uplink_hex(serial_fd, payload_hex)) {
                    std::cerr << "Failed to send uplink for row: " << line << "\n";
                }

                std::this_thread::sleep_for(
                    std::chrono::milliseconds(config_send_delay_ms)
                );
            } catch (const std::exception& row_error) {
                std::cerr << "Skipping bad row [" << line << "]: "
                          << row_error.what() << "\n";
            }
        }

        csv_file.close();
        close(serial_fd);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        if (serial_fd >= 0) {
            close(serial_fd);
        }
        return 1;
    }
}