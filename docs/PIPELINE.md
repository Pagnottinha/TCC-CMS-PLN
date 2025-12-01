# Pipeline de Processamento

Este documento descreve o pipeline completo de processamento do corpus e execução de benchmarks.

## Visão Geral

O pipeline consiste em 4 etapas principais:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Preparação │ -> │ Tokenização │ -> │  Benchmark  │ -> │   Análise   │
│   Corpus    │    │   + PLN     │    │ Estruturas  │    │ Resultados  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
   corpus.py       tokenizar_         benchmark_        
                   corpus.py          estruturas.py     
```

---

## Preparação do Corpus

**Script:** `corpus.py`

### Descrição

Baixa o Corpus Carolina (taxonomia Wikipedia) e divide em 10 partes para processamento.

### Fonte de Dados

- **Dataset:** [carolina-c4ai/corpus-carolina](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina)
- **Taxonomia:** `wik` (Wikipedia em Português)
- **Formato:** Textos completos de artigos

### Processo

1. Download do dataset via HuggingFace Datasets
2. Extração dos textos (`text` field)
3. Embaralhamento com seed reproduzível
4. Divisão em 10 partes iguais
5. Salvamento em `corpus/parte_XX.txt`

### Saída

```
corpus/
├── parte_01.txt    # ~10% dos documentos
├── parte_02.txt
├── ...
└── parte_10.txt
```

Cada arquivo contém um documento por linha (newlines internos são convertidos em espaços).

### Execução

```bash
python corpus.py
```

### Opções Interativas

- `[1] Manter corpus existente` - Pula download se já existe
- `[2] Recriar corpus` - Remove e baixa novamente
- `[3] Sair` - Cancela operação

---

## Tokenização e Normalização

**Script:** `tokenizar_corpus.py`

### Descrição

Processa os documentos através de um pipeline de PLN em 3 fases:

### Fase 1: Limpeza

Remove ruídos do texto:

| Tipo | Exemplos Removidos |
|------|-------------------|
| URLs | `https://...`, `www.exemplo.com` |
| Emails | `usuario@dominio.com` |
| DOIs/ISBNs | `doi:10.xxxx`, `ISBN 978-...` |
| Emoticons | `:)`, `XD`, `^_^` |
| IPs | `192.168.1.1` |
| Menções | `@usuario` |
| Caracteres inválidos | Emojis, símbolos especiais |

### Fase 2: Tokenização e Normalização

Utiliza spaCy com tokenizador customizado:

#### Expansão de Abreviações

```python
'dr' -> 'doutor'
'sr' -> 'senhor'
'prof' -> 'professor'
'etc' -> 'etcétera'
'av' -> 'avenida'
# ... + outras abreviações
```

#### Expansão de Unidades

```python
'km' -> 'quilometro'
'kg' -> 'quilograma'
'km²' -> 'quilometro_quadrado'
# ... + outras unidades
```

#### Tratamento de Clíticos

```python
'fazê-lo' -> 'fazer'
'disse-lhe' -> 'disser'
```

### Fase 3: Lematização e Filtragem

- **Lematização:** Via `simplemma` (lematizador para português)
- **Stopwords:** Remove palavras funcionais (spaCy pt stopwords)
- **Normalização:** Lowercase, remoção de pontuação, tratamento de clíticos e expansão de abreviações e unidades

### Saída

```
corpus/tokenizado/
├── parte_01.txt
├── parte_02.txt
├── ...
└── parte_10.txt
```

Cada linha contém tokens separados por `;`:

```
economia;brasil;crescer;...
politica;governo;federal;...
```

### Execução

```bash
python tokenizar_corpus.py
```

### Configuração

```python
nlp.max_length = 2000000  # Tamanho máximo de documento
batch_size = 2000         # Documentos por lote
n_process = 1             # Processos paralelos
```

---

## Benchmark de Estruturas

**Script:** `benchmark_estruturas.py`

### Descrição

Executa experimentos comparando as estruturas probabilísticas com Counter (ground truth).

### Processo

Para cada parte do corpus:

1. **Criação das estruturas** com configurações definidas
2. **Inserção de tokens:**
   - Palavras individuais
   - Pares de palavras (coocorrências em janela)
3. **Medição de tempos** de inserção
4. **Salvamento** das estruturas
5. **Cálculo do MRE** (Mean Relative Error)
6. **Medição de tempos** de consulta

### Coocorrências

Pares de palavras são extraídos usando janela deslizante:

```python
JANELA = 3  # Configurável via .env

# Documento: "a b c d"
# Pares gerados: (a,b), (a,c), (b,c), (b,d), (c,d)
```

Pares são ordenados alfabeticamente para garantir unicidade:
```python
"palavra1 palavra2"  # palavra1 < palavra2 alfabeticamente
```

### Métricas Coletadas

#### Inserção
- Tempo médio por inserção
- Tempo total
- Desvio padrão
- Mediana
- Tempo mínimo e máximo

#### Consulta
- Tempo médio por consulta
- MRE (palavras, pares, total)
- Desvio padrão
- Mediana
- Tempo mínimo e máximo

### Saída

```
resultados_benchmark/
├── relatorio_completo.json    # Todas as métricas
├── Counter/
│   ├── palavras_parte01.csv   # Ground truth palavras
│   ├── pares_parte01.csv      # Ground truth pares
│   └── ...
├── CMS-CU/
│   ├── parte01.cmscu          # Estrutura serializada
│   └── ...
├── CML8S-CU/
├── CMTS-CU/
├── CL-CU/
├── CL-MU/
└── CML8HS-CU/
```

### Execução

```bash
python benchmark_estruturas.py
```

### Opções Interativas

- `[1] Limpar e executar do zero`
- `[2] Pular inserção, calcular apenas MRE`
- `[3] Exibir relatório existente`
- `[4] Cancelar`

---

## Análise de Resultados

### Análise do Corpus Tokenizado

**Script:** `analisar_corpus.py`

Gera estatísticas sobre o corpus processado:

- Total de documentos
- Total de tokens (palavras + pares)
- Tokens únicos
- Média de tokens por documento

**Saída:** `corpus/relatorio_analise.json`

### Verificação de Lei de Zipf

**Script:** `verificar_zipf.py`

Gera gráficos de distribuição de frequências:

- Gráfico Rank vs Frequência (log-log)
- CCDF (Complementary Cumulative Distribution)

**Saída:** `corpus/grafico_zipf.png`, `corpus/grafico_ccdf.png`

### Comparação de Rankings PPMI

**Script:** `comparar_ranking_ppmi.py`

Compara rankings de coocorrências usando PPMI:

$$PPMI(w_1, w_2) = max(0, log_2 \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)})$$

### Gráficos MRE vs Frequência

**Script:** `gerar_grafico_mre_frequencia.py`

Analisa como o erro varia com a frequência dos elementos.

---

## Configuração

### Variáveis de Ambiente (.env)

```env
# Reprodutibilidade
SEED=42

# Tamanho da janela de coocorrências
JANELA=3

# Configurações de gráficos
FONTSIZE=12
FONTWEIGHT=bold
CORES={"CMS-CU": "#1f77b4", "CML8S-CU": "#ff7f0e", ...}
```

### Estruturas Configuradas

As configurações padrão estão em `factory_estruturas.py`:

```python
CONFIGS_PADRAO = {
    'CMS-CU': {
        'largura': 6_590_290,
        'profundidade': 3,
        'seed': SEED
    },
    # ... outras estruturas
}
```

---

## Métricas de Saída

### relatorio_completo.json

```json
{
  "configuracao": {
    "seed": 42,
    "janela": 3,
    "estruturas": { ... }
  },
  "estatisticas_insercao_totais": {
    "CMS-CU": {
      "media": 1.5e-6,
      "total": 150.5,
      "quantidade": 100000000
    }
  },
  "mre_totais": {
    "CMS-CU": {
      "mre_palavras": 0.0012,
      "mre_pares": 0.0089,
      "mre_total": 0.0051
    }
  },
  "estatisticas_consulta_totais": { ... }
}
```

---

## Fluxo de Dados

```
Carolina Corpus (HuggingFace)
         │
         ▼
┌─────────────────┐
│  corpus.py      │ Download + Split
└────────┬────────┘
         │
         ▼
    corpus/parte_XX.txt (documentos brutos)
         │
         ▼
┌─────────────────┐
│ tokenizar_      │ Limpeza + Tokenização + Lematização
│ corpus.py       │
└────────┬────────┘
         │
         ▼
    corpus/tokenizado/parte_XX.txt (tokens)
         │
         ▼
┌─────────────────┐
│ benchmark_      │ Inserção + MRE + Tempos
│ estruturas.py   │
└────────┬────────┘
         │
         ▼
    resultados_benchmark/
    ├── relatorio_completo.json
    ├── Counter/*.csv
    └── {estrutura}/*.{ext}
         │
         ▼
┌─────────────────┐
│ Scripts de      │ Gráficos + Estatísticas
│ Análise         │
└─────────────────┘
```

---

## Execução Completa

```bash
# 1. Preparar corpus
python corpus.py

# 2. Tokenizar
python tokenizar_corpus.py

# 3. Analisar corpus (opcional)
python analisar_corpus.py
python analise_tokenizado.py

# 4. Benchmark
python benchmark_estruturas.py

# 5. Análises (opcional)
python verificar_zipf.py
python comparar_ranking_ppmi.py
python gerar_grafico_mre_frequencia.py
```

---

## Estimativas de Tempo

| Etapa | Tempo Aproximado |
|-------|------------------|
| Download corpus | 5-15 min |
| Pré-Processamento | 1 hora |
| Benchmark (inserção) | 2-4 horas |
| Benchmark (MRE) | 1-2 horas |
| Ranking PPMI | 1-2 horas |
| Análises | 30-60 min |

*Tempos podem variar significativamente com hardware e tamanho do corpus.*

---

## Troubleshooting

### Erro de Memória

```bash
# Reduza o batch_size em tokenizar_corpus.py
batch_size = 1000  # Padrão: 2000
```

### Erro ao compilar Cython

```bash
# Windows: Instale Visual Studio Build Tools
# Linux: sudo apt install build-essential python3-dev
```

### Corpus não encontrado

```bash
# Verifique se o download completou
ls corpus/parte_*.txt

# Se necessário, force redownload
python corpus.py  # Escolha opção 2
```
