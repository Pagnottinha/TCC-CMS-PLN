"""Vetor de bits baseado em BitFields C++."""

import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef uint8_t BITFIELD_BITSIZE = sizeof(BitField) * 8

cdef class BitVector:
    """Vetor de bits com navegação flat."""
    __slots__ = ()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __cinit__(self, const size_t length):
        """Aloca e inicializa o vetor de bits."""
        if length < 1:
            raise ValueError("Length can't be 0 or negative")

        cdef size_t mask = <size_t>(BITFIELD_BITSIZE - 1)
        cdef size_t align = (mask - ((length - 1) & mask))
        self.length = length + align
        self.size = self.length // BITFIELD_BITSIZE

        self.vector = <BitField *>PyMem_Malloc(self.size * sizeof(BitField))

        cdef size_t bucket
        cdef size_t max_buckets = self.size
        for bucket in range(max_buckets):
            self.vector[bucket].clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __getitem__(self, const size_t index):
        """Retorna o bit no índice especificado."""
        if index >= self.length:
            raise IndexError("Index {} out of range".format(index))

        cdef size_t bucket
        cdef uint8_t bit
        cdef size_t bit_temp

        bucket, bit_temp = divmod(index, BITFIELD_BITSIZE)
        bit = <uint8_t>bit_temp
        return self.vector[bucket].get_bit(bit)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __setitem__(self, const size_t index, const bint flag):
        """Define o bit no índice especificado."""
        if index >= self.length:
            raise IndexError("Index {} out of range".format(index))

        cdef size_t bucket
        cdef uint8_t bit
        cdef size_t bit_temp

        bucket, bit_temp = divmod(index, BITFIELD_BITSIZE)
        bit = <uint8_t>bit_temp
        self.vector[bucket].set_bit(bit, flag)

    def __dealloc__(self):
        PyMem_Free(self.vector)

    def __repr__(self):
        return "<BitVector (size: {}, length: {})>".format(
            self.size,
            self.length
        )

    def __len__(self):
        """Retorna o comprimento do vetor."""
        return self.length

    cpdef size_t sizeof(self):
        """Retorna o tamanho do vetor em bytes."""
        return self.size * sizeof(BitField)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef size_t count(self):
        """Conta o número de bits setados."""
        cdef size_t num_of_bits = 0

        cdef size_t bucket
        cdef size_t max_buckets = self.size
        for bucket in range(max_buckets):
            num_of_bits += self.vector[bucket].count()

        return num_of_bits

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void salvar(self, object file_handle) except *:
        """Salva o vetor em arquivo binário."""
        import struct
        
        cdef size_t bucket
        for bucket in range(self.size):
            file_handle.write(struct.pack('=B', self.vector[bucket].get_field()))
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void carregar(self, object file_handle) except *:
        """Carrega o vetor de arquivo binário."""
        import struct
        
        cdef size_t bucket
        cdef bytes data
        for bucket in range(self.size):
            data = file_handle.read(1)  # uint8_t = 1 byte
            self.vector[bucket].set_field(struct.unpack('=B', data)[0])