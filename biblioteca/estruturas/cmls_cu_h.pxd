from libc.stdint cimport uint64_t, uint32_t, uint8_t


cdef class CMLS8CUH:
    cdef uint8_t _VALOR_MAXIMO_CONTADOR_LOGARITMICO
    cdef uint32_t _VALOR_MAXIMO_CONTADOR_LINEAR

    cdef uint32_t largura
    cdef uint8_t profundidade
    cdef uint8_t camadas_lineares
    cdef double base

    cdef uint64_t _tamanho_linear
    cdef uint64_t _tamanho_logaritmico
    cdef uint64_t _seed
    cdef uint8_t[:] _contador_logaritmico
    cdef uint32_t[:] _contador_linear

    cpdef void incrementar(self, object elemento) except *
    cpdef double estimar(self, object elemento) except *
    cpdef size_t sizeof(self)
    cpdef dict memoria(self)
    cpdef void salvar(self, str nome_arquivo) except *

    cdef bint _incrementar_contador_linear(self, const uint64_t indice)
    cdef bint _incrementar_contador_logaritmico(self, const uint64_t indice)
    cdef uint64_t _hash(self, object elemento, uint64_t seed)