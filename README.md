# TCC-CMS-PLN

**Estruturas de Dados Probabilísticas para Processamento de Linguagem Natural**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cython](https://img.shields.io/badge/Cython-3.1+-orange.svg)](https://cython.org/)

## Sobre o Projeto

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC) que investiga a aplicação de **estruturas de dados probabilísticas** (sketches) para tarefas de Processamento de Linguagem Natural.

O objetivo principal é caracterizar, para tarefas específicas de PLN em português (contagem de frequência de palavras e pares de palavras, e ranking de PPMI), o trade-off entre precisão de contagem, complexidade de implementação e tempo de processamento oferecido pelas variantes CMS, CMTS, Count-Less e CMLS, identificando em quais situações cada uma delas é mais adequada.

Mais detalhes presentes no arquivo ``TCC-GUSTAVO-PAGNOTTA-FARIA.pdf``.

## Estrutura do Projeto

```
TCC-CMS-PLN/
├── biblioteca/              # Implementações das estruturas (Cython)
│   ├── estruturas/          # CMS-CU, CML8S-CU, CMTS, CL, CML8HS-CU
│   └── helpers/             # Hash (CityHash) e BitVector
├── corpus/                  # Corpus processado
│   └── tokenizado/          # Partes tokenizadas
├── analise_resultados/      # Resultados das análises (gráficos)
├── resultados_benchmark/    # Resultados do benchmark
├── main.py                  # Script principal
├── corpus.py                # Preparação do corpus
├── tokenizar_corpus.py      # Pipeline de tokenização
├── benchmark_estruturas.py  # Execução de benchmarks
├── factory_estruturas.py    # Factory para criação de estruturas
└── docs/
    ├── ESTRUTURAS.md        # Documentação das estruturas
    └── PIPELINE.md          # Documentação do pipeline
```

## Instalação

### Pré-requisitos

- Python 3.12+
- Compilador C/C++ (Visual Studio Build Tools no Windows, GCC no Linux)
- Git

### Passos

1. **Clone o repositório:**
```bash
git clone https://github.com/Pagnottinha/TCC-CMS-PLN.git
cd TCC-CMS-PLN
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Compile as extensões Cython:**
```bash
python setup.py build_ext --inplace
```

5. **Configure as variáveis de ambiente:**
```bash
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

## Uso

1. **Preparar o corpus:**
```bash
python corpus.py
```

2. **Tokenizar o corpus:**
```bash
python tokenizar_corpus.py
```

3. **Analisar corpus tokenizado (opcional):**
```bash
python analise_tokenizado.py
```

Etapa realizada para obter a média de tokens por parte. Sendo assim, com base na média de tokens por parte é possível substituir os parâmetros das estruturas em `factory_estruturas.py`.

4. **Executar benchmark:**
```bash
python benchmark_estruturas.py
```

5. **Analisar resultados:**

- analisar_corpus.py: análise do pré-processamento realizado no corpus;
- verificar_zipf.py: análise da lei de zipf no corpus pré-processado;
- gerar_grafico_mre_frequencia.py: análise sobre os MRE de cada frequência dos tokens em cada estrutura;
- comparar_ranking_ppmi.py: comparação das estruturas sobre o ranking de PPMI;

## Documentação Detalhada

- **[ESTRUTURAS.md](docs/ESTRUTURAS.md)** - Documentação das estruturas probabilísticas implementadas
- **[PIPELINE.md](docs/PIPELINE.md)** - Documentação do pipeline de processamento

## Estruturas Implementadas

| Estrutura | Descrição | Características |
|-----------|-----------|-----------------|
| **CMS-CU** | Count-Min Sketch com Conservative Update | Baseline, contadores 32-bit |
| **CML8S-CU** | Count-Min-Log Sketch | Contadores logarítmicos 8-bit |
| **CML8HS-CU** | CML8S com camadas lineares | Híbrido linear/logarítmico |
| **CL-CU/MU** | Count-Less | Camadas hierárquicas expansíveis |
| **CMTS-CU** | Count-Min Tree Sketch | Árvore de bits compacta |

## Configuração

Variáveis de ambiente (`.env`):

```env
SEED=42                    # Semente para reprodutibilidade
JANELA=3                   # Tamanho da janela de coocorrências
```

## Resultados

Os resultados dos benchmarks são salvos em `resultados_benchmark/`:

- `relatorio_completo.json` - Métricas detalhadas
- `Counter/` - Contagens exatas (ground truth)
- `{estrutura}/` - Estruturas salvas por parte

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
