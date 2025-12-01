# Estruturas de Dados Probabilísticas

Este documento descreve as estruturas de dados probabilísticas implementadas neste projeto para contagem de frequências.

## Visão Geral

Todas as estruturas implementadas são variantes do **Count-Min Sketch (CMS)**, uma estrutura de dados probabilística que permite estimar a frequência de elementos em um stream de dados usando memória sublinear.

### Características Comuns

- **Operações principais:** `incrementar(elemento)` e `estimar(elemento)`
- **Trade-off:** Menos memória → maior erro de estimativa
- **Hash:** Utilizam CityHash 64-bit para hashing uniforme
- **Serialização:** Todas suportam `salvar()` e `carregar()`

---

## Estruturas Implementadas

### 1. CMS-CU (Count-Min Sketch com Conservative Update)

**Arquivo:** `biblioteca/estruturas/cms_cu.pyx`

A estrutura baseline. É uma matriz de contadores onde cada elemento é mapeado para uma posição em cada linha usando funções hash independentes.

#### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `largura` | int | Número de contadores por linha |
| `profundidade` | int | Número de linhas (funções hash) |
| `seed` | int | Semente para funções hash |

#### Conservative Update

Em vez de incrementar todos os contadores, incrementa apenas aqueles com valor mínimo. Isso reduz a superestimação.

#### Uso

```python
from biblioteca.estruturas import CMSCU

# Criar estrutura
cms = CMSCU(largura=10000, profundidade=4, seed=42)

# Inserir elementos
cms.incrementar("palavra")
cms.incrementar("palavra")

# Estimar frequência
freq = cms.estimar("palavra")  # Retorna ~2

# Verificar uso de memória
print(f"Tamanho: {cms.sizeof()} bytes")

# Salvar/Carregar
cms.salvar("meu_sketch")
cms_carregado = CMSCU.carregar("meu_sketch")
```

#### Complexidade de Memória

$$M = w \times d \times 4 \text{ bytes}$$

Onde $w$ = largura, $d$ = profundidade, 4 bytes por contador (uint32).

---

### 2. CML8S-CU (Count-Min-Log 8-bit Sketch)

**Arquivo:** `biblioteca/estruturas/cml8s_cu.pyx`

Variante com contadores logarítmicos de 8 bits, permitindo representar valores muito maiores com menos memória.

#### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `largura` | int | Número de contadores por linha |
| `profundidade` | int | Número de linhas |
| `base` | float | Base do logaritmo (ex: 1.08) |
| `seed` | int | Semente para funções hash |


#### Uso

```python
from biblioteca.estruturas import CML8SCU

cms = CML8SCU(largura=30000, profundidade=4, base=1.08, seed=42)
cms.incrementar("elemento")
freq = cms.estimar("elemento")  # Retorna valor aproximado
```

---

### 3. CML8HS-CU (Count-Min-Log 8-bit Hybrid Sketch)

**Arquivo:** `biblioteca/estruturas/cml8hs_cu.pyx`

Versão híbrida que usa contadores lineares para as primeiras camadas e logarítmicos para as demais.

#### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `largura` | int | Número de contadores por linha |
| `profundidade` | int | Número de linhas |
| `camadas_lineares` | int | Quantas camadas usam contadores exatos |
| `base` | float | Base do logaritmo para camadas log |
| `seed` | int | Semente para funções hash |

#### Uso

```python
from biblioteca.estruturas import CML8HSCU

cms = CML8HSCU(
    largura=10000, 
    profundidade=4, 
    camadas_lineares=1,  # Primeira camada exata
    base=1.08, 
    seed=42
)
```

---

### 4. CL (Count-Less Sketch)

**Arquivo:** `biblioteca/estruturas/cl.pyx`

Estrutura hierárquica com camadas expansíveis. Cada camada sucessiva tem largura maior e bits menores por contador.

#### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `largura` | int | Largura base (primeira camada) |
| `profundidade` | int | Número de camadas |
| `expansao` | int | Fator de expansão entre camadas |
| `modo` | int | 0 = Conservative Update, 1 = Minimal Update |
| `seed` | int | Semente para funções hash |

#### Estrutura Hierárquica

```
Camada 0: largura × 32 bits/contador
Camada 1: largura × expansão × 16 bits/contador  
Camada 2: largura × expansão² × 8 bits/contador
Camada 3: largura × expansão³ × 4 bits/contador
```

#### Modos de Atualização

- **Modo 0 (Conservative Update):** Incrementa apenas os mínimos em todas as camadas
- **Modo 1 (Minimal Update):** Propaga incrementos da última para a primeira camada

#### Uso

```python
from biblioteca.estruturas import CL

# Conservative Update
cl_cu = CL(largura=5000, profundidade=4, expansao=4, modo=0, seed=42)

# Minimal Update
cl_mu = CL(largura=5000, profundidade=4, expansao=4, modo=1, seed=42)
```

---

### 5. CMTS (Count-Min Tree Sketch)

**Arquivo:** `biblioteca/estruturas/cmts.pyx`

Estrutura inovadora que usa árvores de bits para representar contadores de forma ultra-compacta. Cada contador é codificado em uma árvore binária.

#### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `largura` | int | Número de árvores por linha |
| `profundidade` | int | Número de linhas |
| `base_arvore` | int | Número de folhas por árvore |
| `seed` | int | Semente para funções hash |

#### Estrutura da Árvore

Cada árvore tem:
- `base_arvore` folhas (elementos contáveis)
- Cada nó usa 2 bits: 1 counting + 1 barrier
- Valores grandes "sobem" a árvore via spire array

#### Uso

```python
from biblioteca.estruturas import CMTS

cmts = CMTS(largura=5000, profundidade=4, base_arvore=128, seed=42)
cmts.incrementar("elemento")
freq = cmts.estimar("elemento")
```

---

## Factory Pattern

Para facilitar a criação de estruturas, use o módulo `factory_estruturas.py`:

```python
from factory_estruturas import (
    criar_estrutura,
    criar_todas_estruturas,
    carregar_estrutura,
    listar_estruturas_disponiveis
)

# Criar com configuração padrão
cms = criar_estrutura('CMS-CU')

# Criar com configuração customizada
cms = criar_estrutura('CMS-CU', {
    'largura': 50000,
    'profundidade': 5,
    'seed': 123
})

# Criar todas as estruturas
estruturas = criar_todas_estruturas()

# Carregar de arquivo
cms = carregar_estrutura('CMS-CU', 'caminho/arquivo')

# Ver estruturas disponíveis
listar_estruturas_disponiveis()
```
