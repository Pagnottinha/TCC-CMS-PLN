import cython

from cython.view cimport array
from libc.math cimport ceil, log, M_E
from libc.stdint cimport uint64_t, uint32_t, uint8_t, UINT32_MAX
from libc.stddef cimport size_t

from biblioteca.helpers.hash.city cimport cityhash_64bit


cdef class CMSCU:
    """
    Exemplo
    -------

    >>> from biblioteca.estruturas import CMSCU

    >>> cms = CMSCU(2000, 4, 25)
    >>> cms.add("ola")
    >>> cms.frequency("ola")

    Nota
    -----
    Essa implementação utiliza contadores de 32-bits, dessa maneira
    o valor máximo é (2^{33} - 1).

    Atributos
    ----------
    largura : :obj:`int`
        Largura do Sketch, sendo o número de contadores no vetor.
    profundidade : :obj:`int`
        Profundidade do Sketch, sendo a quantidade de vetores.
    seed : :obj:`int`
        Semente utilizada para funções hash

    """

    @cython.cdivision(True)
    def __cinit__(self, const uint32_t largura, const uint8_t profundidade, const uint64_t seed):
        """Cria o Sketch com largura, profundidade

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
        self._seed = seed

        self._tamanho = self.largura * self.profundidade

        self._VALOR_MAXIMO_CONTADOR = UINT32_MAX
        self._contador = array(shape=(self._tamanho,), itemsize=sizeof(uint32_t), format='=I')

        cdef uint64_t indice
        for indice in range(self._tamanho):
            self._contador[indice] = 0

    @classmethod
    def criar_apartir_erro_esperado(self, const float desvio, const float erro, const uint64_t seed):
        """Crie um sketch a partir do desvio de frequência esperado e da probabilidade de erro.

        Parâmetros
        ----------
        desvio : float
            O erro ε ao responder a consulta específica.
            Por exemplo, se esperamos 10^7 elementos e permitimos
            um superestimado fixo de 10, o desvio é 10/10^7 = 10^{-6}.
        erro : float
            O erro padrão δ (0 < erro < 1).

        Erros
        ------
        ValueError
            Se `desvio` for menor que 10^{-10}.
        ValueError
            Se `erro` não estiver no intervalo (0, 1).

        """
        if desvio <= 0.0000000001:
            raise ValueError("Taxa de desvio muito pequena. Tente uma taxa maior.")

        if erro <= 0 or erro >= 1:
            raise ValueError("Taxa de erro rate deve estar entre (0, 1)")

        cdef uint8_t profundidade = <uint8_t > (ceil(-log(erro)))
        cdef uint32_t largura = <uint32_t > (ceil(M_E / desvio))

        return self(max(1, largura), max(1, profundidade), seed)

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
        cdef uint64_t indice_elemento
        cdef uint64_t indice

        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t> (h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t> ((h >> 32) & 0xffffffff)

        # Aqui poderia ser chamado a função de estimar, mas dessa maneira
        # seria utilizada a função de hash duas vezes
        cdef uint32_t minimo = self._VALOR_MAXIMO_CONTADOR
        for i in range(self.profundidade):
            indice_elemento = (h1 + i * h2) % self.largura
            indice = i * self.largura + indice_elemento
            if self._contador[indice] < minimo:
                minimo = self._contador[indice]


        for i in range(self.profundidade):
            indice_elemento = (h1 + i * h2) % self.largura
            indice = i * self.largura + indice_elemento

            if self._contador[indice] == minimo:
                self._incrementar_contador(indice)
            
            


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef uint32_t estimar(self, object elemento) except *:
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
        cdef uint32_t indice_elemento
        
        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t> (h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t> ((h >> 32) & 0xffffffff)

        cdef uint64_t indice
        cdef uint32_t frequencia = self._VALOR_MAXIMO_CONTADOR
        for i in range(self.profundidade):
            indice_elemento = (h1 + i * h2) % self.largura
            indice = i * self.largura + indice_elemento
            if self._contador[indice] < frequencia:
                frequencia = self._contador[indice]
        return frequencia

    cpdef size_t sizeof(self):
        """Tamanho do Sketch em bytes

        Returns
        -------
        :obj:`int`
           Número de bytes alocados no Sketch

        """
        cdef size_t total = 0
        total += sizeof(uint32_t)  # _VALOR_MAXIMO_CONTADOR
        total += sizeof(uint32_t)  # largura
        total += sizeof(uint8_t)   # profundidade
        total += sizeof(uint64_t)  # _tamanho
        total += sizeof(uint64_t)  # _seed
        total += self._tamanho * sizeof(uint32_t)  # _contador
        return total

    cpdef void salvar(self, str nome_arquivo) except *:
        """Salva o sketch em um arquivo .cmscu

        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (sem extensão)

        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmscu'):
            nome_arquivo += '.cmscu'
        
        with open(nome_arquivo, 'wb') as f:
            # Escrever cabeçalho: largura (4 bytes), profundidade (1 byte), seed (8 bytes)
            f.write(struct.pack('=IBQ', self.largura, self.profundidade, self._seed))
            
            # Escrever array de contadores
            for i in range(self._tamanho):
                f.write(struct.pack('=I', self._contador[i]))

    @staticmethod
    def carregar(str nome_arquivo):
        """Carrega um sketch de um arquivo .cmscu

        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (sem extensão)

        Retorna
        -------
        CMSCU
            Sketch carregado do arquivo

        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmscu'):
            nome_arquivo += '.cmscu'
        
        with open(nome_arquivo, 'rb') as f:
            # Ler cabeçalho
            header = f.read(13)  # 4 + 1 + 8 = 13 bytes
            largura, profundidade, seed = struct.unpack('=IBQ', header)
            
            # Criar novo sketch
            cms = CMSCU(largura, profundidade, seed)
            
            # Ler array de contadores
            for i in range(cms._tamanho):
                data = f.read(4)  # uint32_t = 4 bytes
                cms._contador[i] = struct.unpack('=I', data)[0]
            
            return cms

    def __repr__(self):
        return "<CMSCU ({} x {})>".format(
            self.largura,
            self.profundidade
        )

    def __len__(self):
        """Pegar a largura do filtro

        Returns
        -------
        :obj:`int`
            A largura do filtro

        """
        return self._tamanho

    cpdef dict memoria(self):
        """Retorna informações sobre uso de memória
        
        Retorna
        -------
        :obj:`dict`
            Dicionário com informações de memória em bytes.
        """
        cdef size_t contador_bytes = self._tamanho * sizeof(uint32_t)
        cdef size_t variaveis_bytes = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint64_t) + sizeof(uint64_t)
        cdef size_t total_bytes = contador_bytes + variaveis_bytes
        
        return {
            'contador': contador_bytes,
            'variaveis': variaveis_bytes,
            'total': total_bytes
        }

    def debug(self):
        return self._contador