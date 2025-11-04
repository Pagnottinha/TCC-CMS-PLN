import cython

from cython.view cimport array
from libc.math cimport pow
from libc.stdint cimport uint64_t, uint32_t, uint8_t, UINT8_MAX, UINT32_MAX
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stddef cimport size_t

from biblioteca.helpers.hash.city cimport cityhash_64bit


cdef class CMLS8CUH:
    """Count-Min-Log Sketch Híbrido com Conservative Update
    
    Combinação de contadores lineares (32-bit) e logarítmicos (8-bit).
    As primeiras camadas usam contadores lineares para precisão, enquanto
    as camadas seguintes usam contadores logarítmicos para economia de memória.
    
    Exemplo
    -------
    >>> from biblioteca.estruturas import CMLS8CUH
    >>> sketch = CMLS8CUH(largura=2000, profundidade=4, camadas_lineares=2, base=1.08, seed=42)
    >>> sketch.incrementar("palavra")
    >>> freq = sketch.estimar("palavra")

    Atributos
    ----------
    largura : uint32_t
        Largura do sketch (número de contadores por camada)
    profundidade : uint8_t
        Profundidade do sketch (número total de camadas)
    camadas_lineares : uint8_t
        Número de camadas com contadores lineares (32-bit)
    base : double
        Base logarítmica para codificação dos contadores logarítmicos
    seed : uint64_t
        Semente para funções hash

    """

    @cython.cdivision(True)
    def __cinit__(self, const uint32_t largura, const uint8_t profundidade, const uint8_t camadas_lineares, const double base, const uint64_t seed):
        """Inicializa o Count-Min-Log Sketch Híbrido

        Parâmetros
        ----------
        largura : uint32_t
            Número de contadores por camada
        profundidade : uint8_t
            Número total de camadas (lineares + logarítmicas)
        camadas_lineares : uint8_t
            Número de camadas com contadores lineares de 32-bit
        base : double
            Base logarítmica para codificação (tipicamente 1.08)
        seed : uint64_t
            Semente para funções hash

        Erros
        ------
        ValueError
            Se largura < 1
        ValueError
            Se profundidade < 1
        ValueError
            Se camadas_lineares > profundidade

        """
        if largura < 1:
            raise ValueError("A largura não pode ser menor que zero.")

        if profundidade < 1:
            raise ValueError("A profundidade não pode ser menor que zero.")

        if camadas_lineares < 1:
            raise ValueError("As camadas lineares não podem ser menor que zero.")

        if camadas_lineares > profundidade:
            raise ValueError("Deve possuir menos camadas lineares que profundidade.")

        self.largura = largura
        self.profundidade = profundidade
        self.base = base
        self._seed = seed
        self.camadas_lineares = camadas_lineares

        srand(<uint32_t> seed)

        self._tamanho_linear = self.largura * camadas_lineares
        self._tamanho_logaritmico = self.largura * (self.profundidade - camadas_lineares) * 4

        self._VALOR_MAXIMO_CONTADOR_LOGARITMICO = UINT8_MAX
        self._VALOR_MAXIMO_CONTADOR_LINEAR = UINT32_MAX
        self._contador_linear = array(shape=(self._tamanho_linear,), itemsize=sizeof(uint32_t), format='=I')
        self._contador_logaritmico = array(shape=(self._tamanho_logaritmico,), itemsize=sizeof(uint8_t), format='B')

        cdef uint64_t indice
        for indice in range(self._tamanho_linear):
            self._contador_linear[indice] = 0
        for indice in range(self._tamanho_logaritmico):
            self._contador_logaritmico[indice] = 0

    cdef uint64_t _hash(self, object key, uint64_t seed):
        return cityhash_64bit(key, seed)

    def __dealloc__(self):
        pass

    cdef bint _incrementar_contador_linear(self, const uint64_t indice):
        """Incrementa contador linear se não estiver no limite
        
        Parâmetros
        ----------
        indice : uint64_t
            Índice do contador linear a incrementar
            
        Retorna
        -------
        bint
            True se incrementou, False se atingiu o limite

        """
        if self._contador_linear[indice] < self._VALOR_MAXIMO_CONTADOR_LINEAR:
            self._contador_linear[indice] += 1
            return True
        return False

    cdef bint _incrementar_contador_logaritmico(self, const uint64_t indice):
        """Incrementa contador logarítmico se não estiver no limite
        
        Parâmetros
        ----------
        indice : uint64_t
            Índice do contador logarítmico a incrementar
            
        Retorna
        -------
        bint
            True se incrementou, False se atingiu o limite

        """
        if self._contador_logaritmico[indice] < self._VALOR_MAXIMO_CONTADOR_LOGARITMICO:
            self._contador_logaritmico[indice] += 1
            return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void incrementar(self, object elemento) except *:
        """Incrementa contador do elemento com Conservative Update
        
        Usa Conservative Update para camadas lineares e incremento
        probabilístico (1/base^min) para camadas logarítmicas.

        Parâmetros
        ----------
        elemento : object
            Elemento a ser contabilizado

        """
        cdef uint8_t i
        cdef uint32_t indice_elemento
        cdef uint64_t indice

        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t> (h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t> ((h >> 32) & 0xffffffff)

        # Aqui poderia ser chamado a função de estimar, mas dessa maneira
        # seria utilizada a função de hash duas vezes
        cdef uint32_t minimo_linear = self._VALOR_MAXIMO_CONTADOR_LINEAR
        cdef uint8_t minimo_logaritmico = self._VALOR_MAXIMO_CONTADOR_LOGARITMICO
        for i in range(self.profundidade):
            if i < self.camadas_lineares:
                indice_elemento = (h1 + i * h2) % self.largura
                indice = i * self.largura + indice_elemento
                if self._contador_linear[indice] < minimo_linear:
                    minimo_linear = self._contador_linear[indice]
            else:
                indice_elemento = (h1 + i * h2) % (self.largura * 4)
                indice = (i - self.camadas_lineares) * (self.largura * 4) + indice_elemento
                if self._contador_logaritmico[indice] < minimo_logaritmico:
                    minimo_logaritmico = self._contador_logaritmico[indice]
                

        for i in range(self.camadas_lineares):
            indice_elemento = (h1 + i * h2) % self.largura
            indice = i * self.largura + indice_elemento

            if self._contador_linear[indice] == minimo_linear:
                self._incrementar_contador_linear(indice)

        # Incrementar com probabilidade 1 / base^minimo
        if ((<double>rand()) / (<double>RAND_MAX)) < (1.0 / pow(self.base, minimo_logaritmico)):
            for i in range(self.camadas_lineares, self.profundidade):
                indice_elemento = (h1 + i * h2) % (self.largura * 4)
                indice = (i - self.camadas_lineares) * (self.largura * 4) + indice_elemento

                if self._contador_logaritmico[indice] == minimo_logaritmico:
                    self._incrementar_contador_logaritmico(indice)
            
            

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double estimar(self, object elemento) except *:
        """Estima frequência do elemento
        
        Para camadas lineares, retorna o valor direto.
        Para camadas logarítmicas, decodifica usando (base^c - 1)/(base - 1).

        Parâmetros
        ----------
        elemento : object
            Elemento cuja frequência será estimada

        Retorna
        -------
        double
            Frequência estimada (mínimo entre todas as camadas)

        """
        cdef uint8_t i
        cdef uint64_t indice_elemento
        
        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t> (h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t> ((h >> 32) & 0xffffffff)

        cdef double valor_decodificado

        cdef uint64_t indice
        cdef double minimo = <double>UINT32_MAX
        for i in range(self.profundidade):
            if i < self.camadas_lineares:
                indice_elemento = (h1 + i * h2) % self.largura
                indice = i * self.largura + indice_elemento
                valor_decodificado = <double> self._contador_linear[indice]
            else:
                indice_elemento = (h1 + i * h2) % (self.largura * 4)
                indice = (i - self.camadas_lineares) * (self.largura * 4) + indice_elemento
                valor_decodificado = (pow(self.base, self._contador_logaritmico[indice]) - 1) / (self.base - 1)
                
            if valor_decodificado < minimo:
                minimo = valor_decodificado
        return minimo

    cpdef size_t sizeof(self):
        """Retorna tamanho total em bytes
        
        Retorna
        -------
        size_t
            Número de bytes alocados (arrays + variáveis)

        """
        cdef size_t total = 0
        total += self._tamanho_logaritmico * sizeof(uint8_t)   # _contador_logaritmico
        total += self._tamanho_linear * sizeof(uint32_t)       # _contador_linear
        total += sizeof(uint32_t)  # largura
        total += sizeof(uint8_t)   # profundidade
        total += sizeof(uint8_t)   # camadas_lineares
        total += sizeof(double)    # base
        total += sizeof(uint64_t)  # _tamanho_logaritmico
        total += sizeof(uint64_t)  # _tamanho_linear
        total += sizeof(uint64_t)  # _seed
        total += sizeof(uint8_t)   # _VALOR_MAXIMO_CONTADOR_LOGARITMICO
        total += sizeof(uint32_t)  # _VALOR_MAXIMO_CONTADOR_LINEAR
        return total

    cpdef dict memoria(self):
        """Retorna detalhamento do uso de memória
        
        Retorna
        -------
        dict
            Dicionário com breakdown da memória:
            - contador_linear: bytes dos contadores lineares (32-bit)
            - contador_logaritmico: bytes dos contadores logarítmicos (8-bit)
            - variaveis: bytes das variáveis de instância
            - total: soma total em bytes
        """
        cdef size_t contador_linear_bytes = self._tamanho_linear * sizeof(uint32_t)
        cdef size_t contador_logaritmico_bytes = self._tamanho_logaritmico * sizeof(uint8_t)
        cdef size_t variaveis_bytes = (sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint8_t) + 
                                       sizeof(double) + sizeof(uint64_t) + sizeof(uint64_t) + 
                                       sizeof(uint64_t) + sizeof(uint8_t) + sizeof(uint32_t))
        cdef size_t total_bytes = contador_linear_bytes + contador_logaritmico_bytes + variaveis_bytes
        
        return {
            'contador_linear': contador_linear_bytes,
            'contador_logaritmico': contador_logaritmico_bytes,
            'variaveis': variaveis_bytes,
            'total': total_bytes
        }

    cpdef void salvar(self, str nome_arquivo) except *:
        """Salva sketch em arquivo binário
        
        Formato: cabeçalho (22 bytes) + contadores lineares + contadores logarítmicos

        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (extensão .cmls8cuh adicionada automaticamente)

        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmls8cuh'):
            nome_arquivo += '.cmls8cuh'
        
        with open(nome_arquivo, 'wb') as f:
            # Escrever cabeçalho: largura (4 bytes), profundidade (1 byte), camadas_lineares (1 byte), base (8 bytes), seed (8 bytes)
            f.write(struct.pack('=IBBdQ', self.largura, self.profundidade, self.camadas_lineares, self.base, self._seed))
            
            # Escrever array de contadores
            for i in range(self._tamanho_linear):
                f.write(struct.pack('=I', self._contador_linear[i]))
            for i in range(self._tamanho_logaritmico):
                f.write(struct.pack('=B', self._contador_logaritmico[i]))

    @staticmethod
    def carregar(str nome_arquivo):
        """Carrega sketch de arquivo binário

        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (extensão .cmls8cuh adicionada automaticamente)

        Retorna
        -------
        CMLS8CUH
            Sketch reconstruído com todos os contadores

        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmls8cuh'):
            nome_arquivo += '.cmls8cuh'
        
        with open(nome_arquivo, 'rb') as f:
            # Ler cabeçalho (4 + 1 + 1 + 8 + 8 = 22 bytes)
            header = f.read(22)
            largura, profundidade, camadas_lineares, base, seed = struct.unpack('=IBBdQ', header)
            
            # Criar novo sketch
            cmls = CMLS8CUH(largura, profundidade, camadas_lineares, base, seed)
            
            # Ler array de contadores
            for i in range(cmls._tamanho_linear):
                data = f.read(4)  # uint32_t = 4 bytes
                cmls._contador_linear[i] = struct.unpack('=I', data)[0]
            
            for i in range(cmls._tamanho_logaritmico):
                data = f.read(1)  # uint8_t = 1 byte
                cmls._contador_logaritmico[i] = struct.unpack('=B', data)[0]
            
            return cmls

    def __repr__(self):
        return "<CMLS8CUH ({} x {}) {} base {}>".format(
            self.largura,
            self.profundidade,
            self.base,
            self.camadas_lineares
        )

    def __len__(self):
        """Retorna número total de contadores
        
        Retorna
        -------
        int
            Soma de contadores lineares e logarítmicos

        """
        return self._tamanho_linear + self._tamanho_logaritmico



    def debug(self):
        return self._contador_linear