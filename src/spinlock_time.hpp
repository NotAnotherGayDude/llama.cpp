#pragma once
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#include "../ggml/src/ggml-impl.h"
static const int spinlock_time = 500;

namespace test {

// from
// https://stackoverflow.com/questions/16337610/how-to-know-if-a-type-is-a-specialization-of-stdvector
template <typename, template <typename...> typename> constexpr bool is_specialization_v = false;

template <template <typename...> typename value_type, typename... arg_types>
constexpr bool is_specialization_v<value_type<arg_types...>, value_type> = true;

template <typename value_type> concept time_type = is_specialization_v<value_type, std::chrono::duration>;

template <time_type value_type = std::chrono::nanoseconds> class stop_watch {
  public:
    using hr_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                        std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    static constexpr bool lock_free{ std::atomic<value_type>::is_always_lock_free };
    using time_type = std::conditional_t<lock_free, value_type, uint64_t>;

    stop_watch(uint64_t newTime) noexcept { total_time_units.store(time_type{ newTime }, std::memory_order_release); }

    stop_watch & operator=(stop_watch && other) noexcept {
        if (this != &other) {
            total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
            start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
            std::swap(values, other.values);
        }
        return *this;
    }

    stop_watch(stop_watch && other) noexcept { *this = std::move(other); }

    stop_watch & operator=(const stop_watch & other) noexcept {
        if (this != &other) {
            total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
            start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
            values = other.values;
        }
        return *this;
    }

    stop_watch(const stop_watch & other) noexcept { *this = other; }

    bool has_time_elapsed() noexcept {
        return ((get_current_time() - start_time_units.load(std::memory_order_acquire)) >=
                total_time_units.load(std::memory_order_acquire));
    }

    void add_time() noexcept {
        //std::unique_lock lock{ mutex };
        values.emplace_back(total_time_elapsed());
        //lock.release();
        reset();
    }

    uint64_t get_count() noexcept { return values.size(); }

    uint64_t get_average(time_type newTimeValue = time_type{}) noexcept {
        std::unique_lock lock{ mutex };
        uint64_t         total_time{};
        for (const auto & value : values) {
            total_time += get_value_as_uint(value);
        }
        return total_time / ((values.size() > 0) ? values.size() : 1);
    }

    void reset(time_type newTimeValue = time_type{}) noexcept {
        if (newTimeValue != time_type{}) {
            total_time_units.store(newTimeValue, std::memory_order_release);
        }
        start_time_units.store(get_current_time(), std::memory_order_release);
    }

    uint64_t get_total_wait_time() const noexcept {
        return get_value_as_uint(total_time_units.load(std::memory_order_acquire));
    }

    time_type total_time_elapsed() noexcept {
        return get_current_time() - start_time_units.load(std::memory_order_acquire);
    }

    uint64_t total_time_elapsed_uint64() noexcept {
        return get_value_as_uint(get_current_time()) -
               get_value_as_uint(start_time_units.load(std::memory_order_acquire));
    }

  protected:
    std::atomic<time_type> total_time_units{};
    std::atomic<time_type> start_time_units{};
    std::vector<time_type> values{};
    std::mutex             mutex{};

    time_type get_current_time() {
        if constexpr (lock_free) {
            return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch());
        } else {
            return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch()).count();
        }
    }

    uint64_t get_value_as_uint(time_type time) {
        if constexpr (lock_free) {
            return time.count();
        } else {
            return time;
        }
    }
};
}  // namespace test

template <bool exceptions> class file_saver {
  public:
    file_saver(const std::filesystem::path & path, const void * data, uint64_t size) {
        if (!data || size == 0) {
            if constexpr (exceptions) {
                throw std::runtime_error("Cannot save null or empty data to file: " + path.string());
            } else {
                std::cerr << "Cannot save null or empty data to file: " + path.string() << std::endl;
            }
        }

        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) {
            if constexpr (exceptions) {
                throw std::runtime_error("Failed to open file for writing: " + path.string());
            } else {
                std::cerr << "Failed to open file for writing: " + path.string() << std::endl;
            }
        }

        file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
        if (!file) {
            if constexpr (exceptions) {
                throw std::runtime_error("Failed to write data to file: " + path.string());
            } else {
                std::cerr << "Failed to write data to file: " + path.string() << std::endl;
            }
        }
    }
};

inline bool have_we_serialized{ false };
