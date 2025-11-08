import cython

from cython.view cimport array
from libc.math cimport ceil, log, pow 
from libc.stdint cimport uint64_t, uint32_t, uint8_t, int8_t, UINT32_MAX
from libc.stddef cimport size_t

from biblioteca.helpers.hash.city cimport cityhash_64bit


cdef class CL:
    """Count-Less Sketch: Sketch hierárquico com camadas expansíveis
    
    Implementa um sketch probabilístico com múltiplas camadas, onde cada
    camada sucessiva possui largura crescente e bits reduzidos por contador.
    Suporta dois modos de incremento:
    - Modo 0 (Conservative Update): Incrementa apenas contadores com valor mínimo
    - Modo 1 (Minimal Update): Incrementa da última para primeira camada
    
    Exemplo
    -------

    >>> from biblioteca.estruturas import CL

    >>> # Conservative Update
    >>> cl0 = CL(2000, 4, 4, 0, 42)
    >>> cl0.incrementar("ola")
    >>> cl0.estimar("ola")
    
    >>> # Modo Minimal Update
    >>> cl1 = CL(2000, 4, 4, 1, 42)
    >>> cl1.incrementar("ola")
    >>> cl1.estimar("ola")

    Atributos
    ----------
    largura : :obj:`int`
        Largura base do Sketch (número de contadores na primeira camada).
    profundidade : :obj:`int`
        Profundidade do Sketch (quantidade de camadas).
    expansao : :obj:`int`
        Taxa de expansão entre camadas (padrão: 4). Cada camada i tem
        largura = largura_base * expansao^i.
    modo : :obj:`int`
        Modo de incremento: 0 (Conservative Update) ou 1 (Minimal Update).
    seed : :obj:`int`
        Semente utilizada para funções hash.

    """

    @cython.cdivision(True)
    def __cinit__(self, const uint32_t largura, const uint8_t profundidade, const uint8_t expansao, const uint8_t modo, const uint64_t seed):
        """Cria o Sketch com largura, profundidade, expansão e modo

        Parâmetros
        ----------
        largura : :obj:`int`
            Largura do Sketch, sendo o número de contadores base.
        profundidade : :obj:`int`
            Profundidade do Sketch, sendo a quantidade de camadas.
        expansao : :obj:`int`
            Taxa de expansão do algoritmo, representa quantas vezes uma
            camada tem a mais que a outra de tamanho (padrão: 4).
        modo : :obj:`int`
            Modo de incremento: 0 para Conservative Update (incrementa mínimos),
            1 para incremento em Minimal Update (da última para primeira camada).
        seed : :obj:`int`
            Semente para a função hash.

        Erros
        ------
        ValueError
            Se `largura` é menor que 1.
        ValueError
            Se `profundidade` é menor que 1.
        ValueError
            Se `expansao` é menor que 1.

        """
        if largura < 1:
            raise ValueError("A largura não pode ser menor que 1.")

        if profundidade < 1:
            raise ValueError("A profundidade não pode ser menor que 1.")

        if expansao < 1:
            raise ValueError("A expansão não pode ser menor que 1.")
        
        if modo not in (0, 1):
            raise ValueError("O modo deve ser 0 (Conservative Update) ou 1 (Minimal Update).")

        self.largura = largura
        self.profundidade = profundidade
        self.expansao = expansao
        self.modo = modo
        self._seed = seed

        # Calcular tamanho total
        if self.expansao == 2:
            self._tamanho = self.largura * self.profundidade
        else:
            self._tamanho = <uint64_t>(2 * self.largura * (pow((<double>self.expansao) / 2, self.profundidade) - 1) / (self.expansao - 2))

        self._offsets_camadas = array(shape=(self.profundidade,), itemsize=sizeof(uint64_t), format='=Q')
        self._largura_camada = array(shape=(self.profundidade,), itemsize=sizeof(uint64_t), format='=Q')
        self._maximo_camada = array(shape=(self.profundidade,), itemsize=sizeof(uint32_t), format='=I')

        cdef uint64_t offset = 0
        cdef uint8_t i
        cdef uint64_t palavras_camada
        cdef uint32_t bits_camada = 0
        
        # Construir de cima para baixo (camada 0 = maior, camada d-1 = menor)
        # Camada i consome: w * (r/2)^i palavras de 32 bits
        for i in range(self.profundidade):
            self._offsets_camadas[i] = offset
            
            # Palavras de 32 bits que esta camada consome
            # Camada 0: w * (r/2)^0 = w
            # Camada 1: w * (r/2)^1 = w * r/2
            # Camada 2: w * (r/2)^2 = w * r²/4
            palavras_camada = <uint64_t>(self.largura * pow((<double>self.expansao) / 2, i))
            offset += palavras_camada
            self._largura_camada[i] =  <uint64_t>(self.largura * pow(self.expansao, i))
            bits_camada = self._bits_camada(i)
            self._maximo_camada[i] = UINT32_MAX if bits_camada >= 32 else (1 << bits_camada) - 1

        # Alocar array de contadores
        self._contador = array(shape=(self._tamanho,), itemsize=sizeof(uint32_t), format='=I')

        cdef uint64_t indice
        for indice in range(self._tamanho):
            self._contador[indice] = 0

    cdef uint64_t _hash(self, object key, uint64_t seed):
        return cityhash_64bit(key, seed)

    def __dealloc__(self):
        pass

    @cython.cdivision(True)
    cdef inline uint8_t _bits_camada(self, const uint8_t camada) noexcept nogil:
        """Retorna bits por contador da camada"""
        cdef uint8_t bits = 32 >> camada  # 32, 16, 8, 4...
        return bits

    @cython.cdivision(True)
    cdef uint32_t _ler_contador(self, const uint32_t posicao_logica, const uint8_t camada):
        """Lê contador empacotado (alinhado)"""
        cdef uint8_t bits = self._bits_camada(camada)
        cdef uint64_t offset = self._offsets_camadas[camada]
        
        # Contadores por palavra
        cdef uint8_t contadores_por_palavra = 32 // bits  # 1, 2, 4
        
        # Qual palavra contém este contador
        cdef uint64_t indice_palavra = offset + (posicao_logica // contadores_por_palavra)
        
        # Posição dentro da palavra (qual contador)
        cdef uint8_t posicao_na_palavra = (posicao_logica % contadores_por_palavra) * bits
        
        # Ler e extrair o valor
        cdef uint32_t palavra = self._contador[indice_palavra]
        cdef uint32_t mascara
        
        if bits >= 32:
            mascara = UINT32_MAX
        else:
            mascara = (1 << bits) - 1

        return (palavra >> posicao_na_palavra) & mascara

    @cython.cdivision(True)
    cdef void _escrever_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor):
        """Escreve contador empacotado (alinhado)"""
        cdef uint8_t bits = self._bits_camada(camada)
        cdef uint64_t offset = self._offsets_camadas[camada]
        
        # Para 32 bits: acesso direto à palavra completa
        if bits >= 32:
            self._contador[offset + posicao_logica] = valor
            return
        
        # Para < 32 bits: empacotamento
        cdef uint8_t contadores_por_palavra = 32 // bits
        cdef uint64_t indice_palavra = offset + (posicao_logica // contadores_por_palavra)
        cdef uint8_t posicao_na_palavra = (posicao_logica % contadores_por_palavra) * bits
        
        cdef uint32_t mascara = ((1 << bits) - 1) << posicao_na_palavra
        cdef uint32_t valor_limpo = (valor & ((1 << bits) - 1)) << posicao_na_palavra
        
        self._contador[indice_palavra] = (self._contador[indice_palavra] & ~mascara) | valor_limpo


    cdef bint _incrementar_contador(self, const uint32_t posicao_logica, const uint8_t camada, const uint32_t valor, const uint32_t maximo):
        """Incrementa o contador se não passar do limite

        Parâmetros
        ----------
        posicao_logica : uint32_t
            A posição lógica do contador
        camada : uint8_t
            A camada do contador
        valor : uint32_t
            O valor atual do contador (já calculado)
        maximo : uint32_t
            O valor máximo da camada (já calculado)

        Retorna
        -------
        bint
            True se incrementou, False se atingiu o limite
        """
        if valor < maximo:
            self._escrever_contador(posicao_logica, camada, valor + 1)
            return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void incrementar(self, object elemento) except *:
        """Incrementa o elemento no Sketch
        
        Utiliza um dos dois modos de incremento:
        - Modo 0 (Conservative Update): Incrementa apenas os contadores com valor mínimo
        - Modo 1 (Minimal Update): Incrementa da última camada para a primeira, propagando valores
        
        Parâmetros
        ----------
        elemento : obj
            O elemento a ser adicionado ou atualizado no Sketch
        """
        cdef int8_t i
        cdef uint32_t posicao_logica
        cdef uint32_t largura_camada
        cdef uint32_t maximo_camada
        cdef uint32_t valor

        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t>(h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t>((h >> 32) & 0xffffffff)

        # Encontrar mínimo entre camadas válidas
        cdef uint32_t minimo = UINT32_MAX
        cdef bint tem_valor_valido = False

        # print(f"incrementar() chamado para {elemento}, h1={h1}, h2={h2}")

        # Modo 0: Conservative Update (incrementa apenas os mínimos)
        if self.modo == 0:
        
            for i in range(self.profundidade):
                largura_camada = self._largura_camada[i]
                posicao_logica = (h1 + i * h2) % largura_camada
                valor = self._ler_contador(posicao_logica, i)
                maximo_camada = self._maximo_camada[i]
                
                # print(f"  Camada {i}: largura={largura_camada}, pos={posicao_logica}, valor={valor}, max={maximo_camada}")
                
                if valor < maximo_camada:
                    if valor < minimo:
                        minimo = valor
                    tem_valor_valido = True

            # print(f"  Mínimo encontrado: {minimo}, tem_valor_valido={tem_valor_valido}")

            if not tem_valor_valido:
                return

            # Incrementar apenas os mínimos
            for i in range(self.profundidade):
                largura_camada = self._largura_camada[i]
                posicao_logica = (h1 + i * h2) % largura_camada
                valor = self._ler_contador(posicao_logica, i)
                maximo_camada = self._maximo_camada[i]
                
                if valor < maximo_camada and valor == minimo:
                    self._incrementar_contador(posicao_logica, i, valor, maximo_camada)
            return
        
        for i in range(self.profundidade - 1, -1, -1):
            largura_camada = self._largura_camada[i]
            posicao_logica = (h1 + i * h2) % largura_camada
            valor = self._ler_contador(posicao_logica, i)
            maximo_camada = self._maximo_camada[i]
            
            # print(f"  Camada {i}: largura={largura_camada}, pos={posicao_logica}, valor={valor}, max={maximo_camada}")
            
            if valor < maximo_camada and valor < minimo:
                self._incrementar_contador(posicao_logica, i, valor, maximo_camada)
                minimo = valor + 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef uint32_t estimar(self, object elemento) except *:
        """Estima frequência do elemento ignorando contadores com overflow
        
        Parâmetros
        ----------
        elemento : obj
            O elemento a ser estimado a frequência
            
        Retorna
        -------
        uint32_t
            A frequência estimada do elemento
        """
        cdef uint8_t i
        cdef uint32_t posicao_logica
        cdef uint32_t largura_camada
        cdef uint32_t maximo_camada
        cdef uint32_t valor
        
        cdef uint64_t h = self._hash(elemento, self._seed)
        cdef uint32_t h1 = <uint32_t>(h & 0xffffffff)
        cdef uint32_t h2 = <uint32_t>((h >> 32) & 0xffffffff)

        cdef uint32_t frequencia = UINT32_MAX
        
        for i in range(self.profundidade):
            largura_camada = self._largura_camada[i]
            posicao_logica = (h1 + i * h2) % largura_camada
            valor = self._ler_contador(posicao_logica, i)
            maximo_camada = self._maximo_camada[i]
            
            if valor < maximo_camada and valor < frequencia:
                frequencia = valor
        
        return frequencia

    cpdef size_t sizeof(self):
        """Tamanho do Sketch em bytes

        Returns
        -------
        :obj:`int`
           Número de bytes alocados no Sketch

        """
        cdef size_t total = 0
        total += sizeof(uint32_t)  # largura
        total += sizeof(uint8_t)   # profundidade
        total += sizeof(uint8_t)   # expansao
        total += sizeof(uint8_t)   # modo
        total += sizeof(uint64_t)  # _tamanho
        total += sizeof(uint64_t)  # _seed
        total += self._tamanho * sizeof(uint32_t)  # _contador
        total += self.profundidade * sizeof(uint64_t)  # _offsets_camadas
        total += self.profundidade * sizeof(uint64_t)  # _largura_camada
        total += self.profundidade * sizeof(uint32_t)  # _maximo_camada
        return total

    cpdef dict memoria(self):
        """Retorna informações sobre uso de memória
        
        Retorna
        -------
        :obj:`dict`
            Dicionário com informações de memória em bytes.
        """
        cdef size_t contador_bytes = self._tamanho * sizeof(uint32_t)
        cdef size_t offsets_camadas_bytes = self.profundidade * sizeof(uint64_t)
        cdef size_t largura_camada_bytes = self.profundidade * sizeof(uint64_t)
        cdef size_t maximo_camada_bytes = self.profundidade * sizeof(uint32_t)
        cdef size_t variaveis_bytes = sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint64_t) + sizeof(uint64_t)
        cdef size_t total_bytes = contador_bytes + offsets_camadas_bytes + largura_camada_bytes + maximo_camada_bytes + variaveis_bytes
        
        return {
            'contador': contador_bytes,
            'offsets_camadas': offsets_camadas_bytes,
            'largura_camada': largura_camada_bytes,
            'maximo_camada': maximo_camada_bytes,
            'variaveis': variaveis_bytes,
            'total': total_bytes
        }

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void salvar(self, str nome_arquivo) except *:
        """Salva o sketch em um arquivo .cl
        
        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (extensão .cl adicionada automaticamente)
        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cl'):
            nome_arquivo += '.cl'
        
        with open(nome_arquivo, 'wb') as f:
            # Escrever cabeçalho: largura (4), profundidade (1), expansao (1), modo (1), seed (8)
            f.write(struct.pack('=IBBBQ', self.largura, self.profundidade, self.expansao, self.modo, self._seed))
            
            # Escrever array de contadores
            for i in range(self._tamanho):
                f.write(struct.pack('=I', self._contador[i]))

    @staticmethod
    def carregar(str nome_arquivo):
        """Carrega um sketch de um arquivo .cl
        
        Parâmetros
        ----------
        nome_arquivo : str
            Nome do arquivo (extensão .cl adicionada automaticamente)
            
        Retorna
        -------
        CL
            Sketch carregado do arquivo
        """
        import struct
        
        cdef uint64_t i
        
        if not nome_arquivo.endswith('.cl'):
            nome_arquivo += '.cl'
        
        with open(nome_arquivo, 'rb') as f:
            # Ler cabeçalho: 4 + 1 + 1 + 1 + 8 = 15 bytes
            header = f.read(15)
            largura, profundidade, expansao, modo, seed = struct.unpack('=IBBBQ', header)
            
            # Criar novo sketch
            cms = CL(largura, profundidade, expansao, modo, seed)
            
            # Ler array de contadores
            for i in range(cms._tamanho):
                data = f.read(4)  # uint32_t = 4 bytes
                cms._contador[i] = struct.unpack('=I', data)[0]
            
            return cms

    def __repr__(self):
        modo_str = "CU" if self.modo == 0 else "Minimal Update"
        return "<CL ({} x {} x {} [{}])>".format(
            self.largura,
            self.profundidade,
            self.expansao,
            modo_str
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
        """Retorna informações de debug
        
        Returns
        -------
        dict
            Dicionário com informações de debug
        """
        return {
            'tamanho_total': self._tamanho,
            'offsets': list(self._offsets_camadas),
            'contador': list(self._contador),
        }