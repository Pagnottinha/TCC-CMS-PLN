"""Interface Cython para CityHash 64-bit."""
from libc.stdint cimport uint64_t
from libc.stddef cimport size_t

cdef extern from "src/city.h":
   uint64_t CityHash64WithSeed(const char *s, size_t len, uint64_t seed)

cdef uint64_t cityhash_64bit_bytes(bytes key, uint64_t seed=<uint64_t>42):
    cdef uint64_t hash_value
    hash_value = CityHash64WithSeed(<char*> key, <size_t> len(key), seed)
    return hash_value

cpdef uint64_t cityhash_64bit(object key, uint64_t seed=<uint64_t>42):
    """Calcula o hash CityHash 64-bit do objeto."""

    if isinstance(key, bytes):
        return cityhash_64bit_bytes(key, seed)

    return cityhash_64bit_bytes(repr(key).encode("utf-8"), seed)