from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libc.stddef cimport size_t

cdef class CMSCU:
    cdef uint32_t _VALOR_MAXIMO_CONTADOR

    cdef uint32_t largura
    cdef uint8_t profundidade

    cdef uint64_t _tamanho
    cdef uint64_t _seed
    cdef uint32_t[:] _contador

    cpdef void incrementar(self, object elemento) except *
    cpdef uint32_t estimar(self, object elemento) except *
    cpdef size_t sizeof(self)
    cpdef dict memoria(self)
    cpdef void salvar(self, str nome_arquivo) except *

    cdef bint _incrementar_contador(self, const uint64_t indice)
    cdef uint64_t _hash(self, object elemento, uint64_t seed)