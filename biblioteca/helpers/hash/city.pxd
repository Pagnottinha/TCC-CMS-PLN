from libc.stdint cimport uint64_t

cdef uint64_t cityhash_64bit_bytes(bytes key, uint64_t seed=*)

cpdef uint64_t cityhash_64bit(object key, uint64_t seed=*)