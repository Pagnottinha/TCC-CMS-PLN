import cython

from cython.view cimport array
from libc.math cimport floor, log2, pow
from libc.stdint cimport uint64_t, uint32_t, uint8_t, UINT32_MAX
from libc.stddef cimport size_t

from biblioteca.helpers.hash.city cimport cityhash_64bit
from biblioteca.helpers.storage.bitvector cimport BitVector

# Implementação multiplataforma de count leading zeros
cdef extern from * nogil:
    """
    #ifdef _MSC_VER
    #include <intrin.h>
    static inline int __builtin_clzll(unsigned long long x) {
        unsigned long index;
        if (_BitScanReverse64(&index, x)) {
            return 63 - (int)index;
        }
        return 64;
    }
    #else
    int __builtin_clzll(unsigned long long);
    #endif
    """
    int __builtin_clzll(unsigned long long)

cdef class CMTS:
    """
    Exemplo
    -------

    >>> from biblioteca.estruturas import CMTS

    >>> cmts = CMTS(2000, 4, 16, 25)
    >>> cmts.incrementar("ola")
    >>> cmts.estimar("ola")

    Atributos
    ----------
    largura : :obj:`int`
        Largura do Sketch, sendo o número de contadores no vetor.
    profundidade : :obj:`int`
        Profundidade do Sketch, sendo a quantidade de vetores.
    base_arvore: :obj:`int`
        Base da árvore utilizada na estrutura de contadores.
    seed : :obj:`int`
        Semente utilizada para funções hash

    """

    @cython.cdivision(True)
    def __cinit__(self, const uint32_t largura, const uint8_t profundidade, const uint8_t base_arvore, const uint64_t seed):
        """Cria o Sketch com largura, profundidade e base da árvore

        Parâmetros
        ----------
        largura : :obj:`int`
            Largura do Sketch, sendo o número de contadores.
        profundidade : :obj:`int`
            Profundidade do Sketch, sendo a quantidade de vetores.
        base_arvore: :obj:`int`
            Base da árvore utilizada na estrutura de contadores.
        seed : :obj:`int`
            Semente para a função hash

        Erros
        ------
        ValueError
            Se `largura` é menor que 1.
        ValueError
            Se `profundidade` é menor que 1.
        ValueError
            Se `base_arvore` é menor que 1.

        """
        if largura < 1:
            raise ValueError("A largura não pode ser menor que zero.")

        if profundidade < 1:
            raise ValueError("A profundidade não pode ser menor que zero.")

        if base_arvore < 1:
            raise ValueError("A base da arvore não pode ser menor que zero.")

        # print(f"[CMTS] Iniciando construção: largura={largura}, profundidade={profundidade}, base_arvore={base_arvore}")

        self.largura = largura
        self.profundidade = profundidade
        self.base_arvore = base_arvore
        self._seed = seed
        self._altura_arvore = (<uint8_t>log2(<double>self.base_arvore)) + 1
        self._tamanho_arvore = 2 * (<uint64_t>pow(2, self._altura_arvore) - 1)

        # print(f"[CMTS] Calculado: altura_arvore={self._altura_arvore}, tamanho_arvore={self._tamanho_arvore}")

        self._quantidade_contadores = <uint64_t>(self.largura * self.profundidade)
        self._tamanho_bit = <uint64_t>(self._quantidade_contadores * self._tamanho_arvore)

        # print(f"[CMTS] Quantidade contadores={self._quantidade_contadores}, tamanho_bit={self._tamanho_bit}")

        self._contador = BitVector(self._tamanho_bit)

        # print(f"[CMTS] BitVector criado com sucesso")

        self._spire = array(shape=(self._quantidade_contadores,), itemsize=sizeof(uint32_t), format='=I')

        cdef uint64_t i_spire
        for i_spire in range(self._quantidade_contadores):
            self._spire[i_spire] = 0

        # print(f"[CMTS] Construção concluída com sucesso!")



    cdef uint64_t _hash(self, object key, uint64_t seed):
        return cityhash_64bit(key, seed)

    cdef uint64_t _lsb(self, uint64_t val):
        if val == 0: return 0
        return 64 - __builtin_clzll(val)

    def __dealloc__(self):
        pass

    @cython.cdivision(True)
    cdef uint32_t _ler_contador(self, const uint32_t posicao_logica, const uint8_t camada):
        """Lê o valor de um contador na posição e camada especificadas
        
        Parâmetros
        ----------
        posicao_logica : :obj:`int`
            Posição lógica do contador.
        camada : :obj:`int`
            Camada do contador.
            
        Retorna
        -------
        :obj:`int`
            Valor do contador.
        """
        cdef uint64_t contador = posicao_logica / self.base_arvore
        cdef uint64_t contador_arvore = posicao_logica % self.base_arvore
        
        if contador >= self.largura:
            return 0
        
        cdef uint64_t offset_camada = camada * self.largura * self._tamanho_arvore
        cdef uint64_t offset_arvore = contador * self._tamanho_arvore

        cdef uint64_t i = 0, b = 0, c = 0
        cdef uint64_t offset_camada_arvore = 0
        cdef uint64_t bits_camada = self.base_arvore * 2
        cdef uint64_t tpos = contador_arvore
        cdef uint64_t posicao
        
        for i in range(self._altura_arvore):
            posicao = offset_camada + offset_arvore + offset_camada_arvore + tpos * 2

            if self._contador[posicao + 1] == 0: 
                if self._contador[posicao] == 1:
                    c |= ((<uint64_t>1) << i)
                break

            b += 1

            if self._contador[posicao] == 1:
                c |= ((<uint64_t>1) << i)

            tpos >>= 1
            offset_camada_arvore += bits_camada
            bits_camada >>= 1 

        if b == self._altura_arvore:
            c = (self._spire[contador + self.largura * camada] << b) + c

        return c + 2 * (((<uint64_t> 1) << b) - 1)
        
        

    @cython.cdivision(True)
    cdef void _escrever_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor):
        """Escreve um valor em um contador na posição e camada especificadas
        
        Parâmetros
        ----------
        posicao_logica : :obj:`int`
            Posição lógica do contador.
        camada : :obj:`int`
            Camada do contador.
        valor : :obj:`int`
            Valor a ser escrito.
        """
        cdef uint64_t contador = posicao_logica / self.base_arvore
        cdef uint64_t contador_arvore = posicao_logica % self.base_arvore
        
        if contador >= self.largura:
            return

        cdef uint64_t x = (valor + 2) / 4
        cdef uint64_t valor_lsb = self._lsb(x)

        cdef uint64_t nb = self._altura_arvore if self._altura_arvore < valor_lsb else valor_lsb
        cdef uint64_t nc = valor - 2 * (((<uint64_t>1) << nb) - 1)

        cdef uint64_t offset_camada = camada * self.largura * self._tamanho_arvore
        cdef uint64_t offset_arvore = contador * self._tamanho_arvore

        cdef uint64_t i = 0
        cdef uint64_t offset_camada_arvore = 0
        cdef uint64_t bits_camada = self.base_arvore * 2
        cdef uint64_t tpos = contador_arvore
        cdef uint64_t posicao
        
        for i in range(min(nb + 1, self._altura_arvore)):
            posicao = offset_camada + offset_arvore + offset_camada_arvore + tpos * 2
            
            # Sempre escreve counting
            self._contador[posicao] = (nc % 2) != 0
            
            # Escreve barrier APENAS se i < nb
            if i < nb:
                self._contador[posicao + 1] = 1
            
            nc >>= 1
            tpos >>= 1
            offset_camada_arvore += bits_camada
            bits_camada >>= 1
        
        cdef uint64_t indice_spire = contador + self.largura * camada
        if nb >= self._altura_arvore and self._spire[indice_spire] < nc:
            self._spire[indice_spire] = nc



    cdef bint _incrementar_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor):
        """Incrementa o contador se não passar do limite

        Parâmetros
        ----------
        posicao_logica : :obj:`int`
            Posição lógica do contador.
        camada : :obj:`int`
            Camada do contador.
        valor : :obj:`int`
            Valor atual do contador.

        Retorna
        -------
        :obj:`bool`
            True se incrementou, False se atingiu o limite.
        """
        if valor < UINT32_MAX:
            self._escrever_contador(posicao_logica, camada, valor + 1)
            return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void incrementar(self, object elemento) except *:
        """Incrementa o elemento no Sketch com Minimum Update
        
        Parâmetros
        ----------
        elemento : :obj:`str` ou :obj:`bytes`
            O elemento a ser adicionado ou atualizado no Sketch.
        """
        # print(f"[incrementar] Chamado para elemento={elemento}")
        
        cdef uint8_t i
        cdef uint32_t posicao_logica
        cdef uint32_t valor

        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t>(h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t>((h >> 32) & 0xffffffff)

        # print(f"[incrementar] h1={h1}, h2={h2}")

        # Encontrar mínimo entre camadas válidas
        cdef uint32_t minimo = UINT32_MAX
        cdef bint tem_valor_valido = False
        
        for i in range(self.profundidade):
            posicao_logica = (h1 + i * h2) % (self.largura * self.base_arvore)
            # print(f"[incrementar] Camada {i}: pos_logica={posicao_logica}, largura*base={self.largura * self.base_arvore}")
            valor = self._ler_contador(posicao_logica, i)
            
            # # print(f"  Camada {i}: largura={largura_camada}, pos={posicao_logica}, valor={valor}, max={maximo_camada}")
            
            if valor < minimo:
                minimo = valor
            tem_valor_valido = True

        # # print(f"  Mínimo encontrado: {minimo}, tem_valor_valido={tem_valor_valido}")

        if not tem_valor_valido:
            return

        # Incrementar apenas os mínimos
        for i in range(self.profundidade):
            posicao_logica = (h1 + i * h2) % (self.largura * self.base_arvore)
            valor = self._ler_contador(posicao_logica, i)
            
            if valor == minimo:
                self._incrementar_contador(posicao_logica, i, valor)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef uint32_t estimar(self, object elemento) except *:
        """Estima a frequência do elemento
        
        Parâmetros
        ----------
        elemento : :obj:`str` ou :obj:`bytes`
            O elemento a ser estimado a frequência.
            
        Retorna
        -------
        :obj:`int`
            A frequência estimada do elemento.
        """
        cdef uint8_t i
        cdef uint32_t posicao_logica
        cdef uint32_t valor
        
        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t>(h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t>((h >> 32) & 0xffffffff)

        cdef uint32_t frequencia = UINT32_MAX
        
        for i in range(self.profundidade):
            posicao_logica = (h1 + i * h2) % (self.largura * self.base_arvore)
            valor = self._ler_contador(posicao_logica, i)
            
            if valor < frequencia:
                frequencia = valor
        
        return frequencia

    cpdef size_t sizeof(self):
        """Retorna o tamanho do Sketch em bytes

        Retorna
        -------
        :obj:`int`
            Número de bytes alocados no Sketch.

        """
        cdef size_t total = 0
        total += sizeof(uint32_t)  # largura
        total += sizeof(uint8_t)   # profundidade
        total += sizeof(uint8_t)   # base_arvore
        total += sizeof(uint8_t)   # _altura_arvore
        total += sizeof(uint64_t)  # _tamanho_arvore
        total += sizeof(uint64_t)  # _tamanho_bit
        total += sizeof(uint64_t)  # _quantidade_contadores
        total += sizeof(uint64_t)  # _seed
        total += self._contador.sizeof()  # BitVector
        total += self._quantidade_contadores * sizeof(uint32_t)  # _spire
        return total

    cpdef dict memoria(self):
        """Retorna informações sobre uso de memória
        
        Retorna
        -------
        :obj:`dict`
            Dicionário com informações de memória em bytes.
        """
        cdef size_t bitvector_bytes = self._contador.sizeof()
        cdef size_t spire_bytes = self._quantidade_contadores * sizeof(uint32_t)
        cdef size_t variaveis_bytes = sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint64_t)
        cdef size_t total_bytes = bitvector_bytes + spire_bytes + variaveis_bytes
        
        return {
            'bitvector': bitvector_bytes,
            'spire': spire_bytes,
            'variaveis': variaveis_bytes,
            'total': total_bytes
        }

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void salvar(self, str nome_arquivo) except *:
        """Salva o sketch em um arquivo .cmts
        
        Parâmetros
        ----------
        nome_arquivo : :obj:`str`
            Nome do arquivo (com ou sem extensão).
        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmts'):
            nome_arquivo += '.cmts'
        
        with open(nome_arquivo, 'wb') as f:
            # Escrever cabeçalho: largura (4 bytes), profundidade (1 byte), base_arvore (1 byte), seed (8 bytes)
            f.write(struct.pack('=IBBQ', self.largura, self.profundidade, self.base_arvore, self._seed))
            
            # Escrever BitVector
            self._contador.salvar(f)

            # Escrever array de spire
            for i in range(self._quantidade_contadores):
                f.write(struct.pack('=I', self._spire[i]))

    @staticmethod
    def carregar(str nome_arquivo):
        """Carrega um sketch de um arquivo .cmts
        
        Parâmetros
        ----------
        nome_arquivo : :obj:`str`
            Nome do arquivo (com ou sem extensão).
            
        Retorna
        -------
        CMTS
            Sketch carregado do arquivo.
        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cmts'):
            nome_arquivo += '.cmts'
        
        with open(nome_arquivo, 'rb') as f:
            # Ler cabeçalho: 4 + 1 + 1 + 8 = 14 bytes
            header = f.read(14)
            largura, profundidade, base_arvore, seed = struct.unpack('=IBBQ', header)
            
            # Criar novo sketch
            cms = CMTS(largura, profundidade, base_arvore, seed)
            
            # Carregar BitVector
            cms._contador.carregar(f)

            # Ler array de spire
            for i in range(cms._quantidade_contadores):
                data = f.read(4)  # uint32_t = 4 bytes
                cms._spire[i] = struct.unpack('=I', data)[0]
            
            return cms

    def __repr__(self):
        return "<CMTS ({} x {} x {})>".format(
            self.largura,
            self.profundidade,
            self.base_arvore
        )

    def __len__(self):
        """Retorna a quantidade de contadores no Sketch

        Retorna
        -------
        :obj:`int`
            A quantidade de contadores.

        """
        return self._quantidade_contadores



    def debug(self):
        """Retorna informações de debug
        
        Retorna
        -------
        :obj:`dict`
            Dicionário com informações de debug.
        """
        return {
            'tamanho_total': self._tamanho_bit,
            'contador': list(self._contador),
            'spire': list(self._spire)
        }