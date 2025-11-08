import cython

from cython.view cimport array
from libc.math cimport pow
from libc.stdint cimport uint64_t, uint32_t, uint8_t, UINT8_MAX
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stddef cimport size_t

from biblioteca.helpers.hash.city cimport cityhash_64bit


cdef class CMLS8CU:
    """
    Exemplo
    -------

    >>> from biblioteca.estruturas import CMSCU

    >>> cms = CMLS8CU(2000, 4, 25)
    >>> cms.add("ola")
    >>> cms.frequency("ola")

    Nota
    -----
    Essa implementação utiliza contadores de 8-bits, dessa maneira
    o valor máximo é (2^{9} - 1).

    Atributos
    ----------
    largura : :obj:`int`
        Largura do Sketch, sendo o número de contadores no vetor.
    profundidade : :obj:`int`
        Profundidade do Sketch, sendo a quantidade de vetores.
    seed : :obj:`int`
        Semente utilizada para funções hash
    base : :obj:`int`
        Base uilizada para o algoritmo

    """

    @cython.cdivision(True)
    def __cinit__(self, const uint32_t largura, const uint8_t profundidade, const double base, const uint64_t seed):
        """Cria o Sketch com largura, profundidade, base

        Parâmetros
        ----------
        largura : :obj:`int`
            Largura do Skech, sendo o número de contadores.
        profundidade : :obj:`int`
            Profundidade do Sketch, sendo a quantidade de vetores.
        seed : :obj:`int`
            Semente para a função hash

        Erros
        ------
        ValueError
            Se `largura` é menor que 1.
        ValueError
            Se `profundidade` é menor que 1.

        """
        if largura < 1:
            raise ValueError("A largura não pode ser menor que zero.")

        if profundidade < 1:
            raise ValueError("A profundidade não pode ser menor que zero.")

        self.largura = largura
        self.profundidade = profundidade
        self.base = base
        self._seed = seed

        srand(<uint32_t> seed)

        self._tamanho = self.largura * self.profundidade

        self._VALOR_MAXIMO_CONTADOR = UINT8_MAX
        self._contador = array(shape=(self._tamanho,), itemsize=sizeof(uint8_t), format='B')

        cdef uint64_t indice
        for indice in range(self._tamanho):
            self._contador[indice] = 0

    cdef uint64_t _hash(self, object key, uint64_t seed):
        return cityhash_64bit(key, seed)

    def __dealloc__(self):
        pass

    cdef bint _incrementar_contador(self, const uint64_t indice):
        """Incrementa o contador se não passar do limite

        Parâmetros
        ----------
        indice : obj:`int`
            O indice a ser incrementado

        """
        if self._contador[indice] < self._VALOR_MAXIMO_CONTADOR:
            self._contador[indice] += 1
            return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void incrementar(self, object elemento) except *:
        """Incremetar o elemento no Skecth

        Parâmetros
        ----------
        elemento : obj
            O elemento a ser adicionado ou atualizado no Sketch

        """
        cdef uint8_t i
        cdef uint32_t indice_elemento
        cdef uint64_t indice

        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t> (h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t> ((h >> 32) & 0xffffffff)

        # Aqui poderia ser chamado a função de estimar, mas dessa maneira
        # seria utilizada a função de hash duas vezes
        cdef uint8_t minimo = self._VALOR_MAXIMO_CONTADOR
        for i in range(self.profundidade):
            indice_elemento = (h1 + i * h2) % self.largura
            indice = i * self.largura + indice_elemento
            if self._contador[indice] < minimo:
                minimo = self._contador[indice]

        # Incrementar com probabilidade 1 / base^minimo
        if ((<double>rand()) / (<double>RAND_MAX)) < (1.0 / pow(self.base, minimo)):
            for i in range(self.profundidade):
                indice_elemento = (h1 + i * h2) % self.largura
                indice = i * self.largura + indice_elemento

                if self._contador[indice] == minimo:
                    self._incrementar_contador(indice)
            
            

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double estimar(self, object elemento) except *:
        """Estima frequencia do elemento.

        Parâmetros
        ----------
        elemento : obj
            O elemento a ser estimado a frequência.

        Retorna
        -------
        uint32_t
            A frequência do elemento.

        """
        cdef uint8_t i
        cdef uint64_t indice_elemento
        
        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t> (h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t> ((h >> 32) & 0xffffffff)

        cdef uint64_t indice
        cdef uint8_t frequencia = self._VALOR_MAXIMO_CONTADOR
        for i in range(self.profundidade):
            indice_elemento = (h1 + i * h2) % self.largura
            indice = i * self.largura + indice_elemento
            if self._contador[indice] < frequencia:
                frequencia = self._contador[indice]

        return (pow(self.base, frequencia) - 1) / (self.base - 1)

    cpdef size_t sizeof(self):
        """Retorna o tamanho do Sketch em bytes

        Retorna
        -------
        :obj:`int`
            Número de bytes alocados no Sketch.

        """
        cdef size_t total = 0
        total += sizeof(uint8_t)   # _VALOR_MAXIMO_CONTADOR
        total += sizeof(uint32_t)  # largura
        total += sizeof(uint8_t)   # profundidade
        total += sizeof(double)    # base
        total += sizeof(uint64_t)  # _tamanho
        total += sizeof(uint64_t)  # _seed
        total += self._tamanho * sizeof(uint8_t)  # _contador
        return total

    cpdef dict memoria(self):
        """Retorna informações sobre uso de memória
        
        Retorna
        -------
        :obj:`dict`
            Dicionário com informações de memória em bytes.
        """
        cdef size_t contador_bytes = self._tamanho * sizeof(uint8_t)
        cdef size_t variaveis_bytes = sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint8_t) + sizeof(double) + sizeof(uint64_t) + sizeof(uint64_t)
        cdef size_t total_bytes = contador_bytes + variaveis_bytes
        
        return {
            'contador': contador_bytes,
            'variaveis': variaveis_bytes,
            'total': total_bytes
        }

    cpdef void salvar(self, str nome_arquivo) except *:
        """Salva o sketch em um arquivo .cmls8cu

        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (sem extensão)

        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmls8cu'):
            nome_arquivo += '.cmls8cu'
        
        with open(nome_arquivo, 'wb') as f:
            # Escrever cabeçalho: largura (4 bytes), profundidade (1 byte), base (8 bytes), seed (8 bytes)
            f.write(struct.pack('=IBdQ', self.largura, self.profundidade, self.base, self._seed))
            
            # Escrever array de contadores
            for i in range(self._tamanho):
                f.write(struct.pack('=B', self._contador[i]))

    @staticmethod
    def carregar(str nome_arquivo):
        """Carrega um sketch de um arquivo .cmls8cu

        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (sem extensão)

        Retorna
        -------
        CMLS8CU
            Sketch carregado do arquivo

        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmls8cu'):
            nome_arquivo += '.cmls8cu'
        
        with open(nome_arquivo, 'rb') as f:
            # Ler cabeçalho (4 + 1 + 8 + 8 = 21 bytes)
            header = f.read(21)
            largura, profundidade, base, seed = struct.unpack('=IBdQ', header)
            
            # Criar novo sketch
            cmls = CMLS8CU(largura, profundidade, base, seed)
            
            # Ler array de contadores
            for i in range(cmls._tamanho):
                data = f.read(1)  # uint8_t = 1 byte
                cmls._contador[i] = struct.unpack('=B', data)[0]
            
            return cmls

    def __repr__(self):
        return "<CMLS8CU ({} x {}) {} base>".format(
            self.largura,
            self.profundidade,
            self.base
        )

    def __len__(self):
        """Pegar a largura do filtro

        Returns
        -------
        :obj:`int`
            A largura do filtro

        """
        return self._tamanho



    def debug(self):
        return self._contador