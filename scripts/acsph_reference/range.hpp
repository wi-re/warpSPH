// Minimal replacement for the absent util::lang range helper.
// Only positive-stride integer iteration is used by dynGPUSPH.
#pragma once
namespace util { namespace lang {
template<class T> struct range_proxy {
    struct iterator {
        T value, stride;
        __host__ __device__ T operator*() const { return value; }
        __host__ __device__ iterator& operator++() { value += stride; return *this; }
        __host__ __device__ bool operator!=(const iterator& end) const { return value < end.value; }
    };
    struct step_range_proxy {
        T first, last, stride;
        __host__ __device__ iterator begin() const { return {first, stride}; }
        __host__ __device__ iterator end() const { return {last, stride}; }
    };
    T first, last;
    __host__ __device__ step_range_proxy step(T stride) const { return {first,last,stride}; }
};
template<class T> __host__ __device__ range_proxy<T> range(T first, T last) { return {first,last}; }
}}
