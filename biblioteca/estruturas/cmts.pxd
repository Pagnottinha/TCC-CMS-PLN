from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libc.stddef cimport size_t

from biblioteca.helpers.storage.bitvector cimport BitVector


cdef class CMTS:
    cdef uint32_t largura
    cdef uint8_t profundidade
    cdef uint8_t base_arvore

    cdef uint8_t _altura_arvore
    cdef uint64_t _tamanho_arvore
    cdef uint64_t _tamanho_bit
    cdef uint64_t _quantidade_contadores
    cdef uint64_t _seed
    cdef BitVector _contador
    cdef uint32_t [:] _spire

    cdef uint64_t[:] _offsets_camadas
    cdef uint64_t[:] _offsets_arvores
    cdef uint32_t[:] _offsets_camada_arvore
    
    cdef uint32_t _ler_contador(self, const uint32_t posicao_logica, const uint8_t camada)
    cdef void _escrever_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor)
    cdef uint64_t _lsb(self, uint64_t val)

    cpdef void incrementar(self, object elemento) except *
    cpdef uint32_t estimar(self, object elemento) except *
    cpdef size_t sizeof(self)
    cpdef dict memoria(self)
    cpdef void salvar(self, str nome_arquivo) except *

    cdef bint _incrementar_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor)
    cdef uint64_t _hash(self, object elemento, uint64_t seed)