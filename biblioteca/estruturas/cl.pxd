from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libc.stddef cimport size_t

cdef class CL:
    cdef uint32_t largura
    cdef uint8_t profundidade
    cdef uint8_t expansao
    cdef uint8_t modo 

    cdef uint64_t _tamanho
    cdef uint64_t _seed
    cdef uint32_t[:] _contador

    cdef uint64_t[:] _offsets_camadas
    cdef uint64_t[:] _largura_camada
    cdef uint32_t[:] _maximo_camada
    
    cdef inline uint8_t _bits_camada(self, const uint8_t camada) noexcept nogil
    cdef uint32_t _ler_contador(self, const uint32_t posicao_logica, const uint8_t camada)
    cdef void _escrever_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor)

    cpdef void incrementar(self, object elemento) except *
    cpdef uint32_t estimar(self, object elemento) except *
    cpdef size_t sizeof(self)
    cpdef dict memoria(self)
    cpdef void salvar(self, str nome_arquivo) except *

    cdef bint _incrementar_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor, const uint32_t maximo)
    cdef uint64_t _hash(self, object elemento, uint64_t seed)