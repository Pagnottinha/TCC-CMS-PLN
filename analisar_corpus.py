"""
Análise comparativa de estratégias de tokenização.

Compara três estratégias:
1. Mínima: Limpeza, tokenização e minúsculas.
2. Mínima + SW: Anterior com remoção de stopwords.
3. Completa: Lematizada e normalizada.
"""
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Importa apenas a função de limpeza do script original
from tokenizar_corpus import limpar_texto

# Cria um pipeline spaCy simples apenas para tokenização
nlp_simples = spacy.blank("pt")
nlp_simples.max_length = 2000000


CORES = json.loads(os.getenv('CORES', '{}'))
LABEL_FONTSIZE = int(os.getenv('LABEL_FONTSIZE', 16))
LEGEND_FONTSIZE = int(os.getenv('LEGEND_FONTSIZE', 16))
TICK_LABEL_FONTSIZE = int(os.getenv('TICK_LABEL_FONTSIZE', 14))
BAR_LABEL_FONTSIZE = int(os.getenv('BAR_LABEL_FONTSIZE', 11)) # New constant for bar labels
LINEWIDTH = float(os.getenv('LINEWIDTH', 2.5))
FONTWEIGHT = os.getenv('FONTWEIGHT', 'bold')

# Default colors for specific strategies if not provided in .env
DEFAULT_STRATEGY_COLORS = {
    "Mínima": "#1f77b4",
    "Mínima+SW": "#ff7f0e",
    "Completa": "#2ca02c"
}
# Merge default colors with any user-defined colors from .env
STRATEGY_COLORS = {**DEFAULT_STRATEGY_COLORS, **CORES}


def analisar_estrategias_basicas(partes_originais):
    """Analisa o corpus com estratégias Mínima e Mínima + SW."""
    print("\nAnalisando com as estratégias básicas (Mínima e Mínima + SW)...")

    stats_min_gerais = {'num_documentos': 0, 'tokens_totais': 0, 'tokens_unicos': set(), 'tamanho_bytes': 0}
    stats_min_partes = []

    stats_sw_gerais = {'num_documentos': 0, 'tokens_totais': 0, 'tokens_unicos': set(), 'tamanho_bytes': 0}
    stats_sw_partes = []

    for parte_path in tqdm(partes_originais, desc="Estratégias Básicas"):
        tamanho_bytes = parte_path.stat().st_size

        stats_min_parte = {'nome': parte_path.name, 'num_documentos': 0, 'tokens_totais': 0, 'tokens_unicos': set(), 'tamanho_bytes': tamanho_bytes}
        stats_sw_parte = {'nome': parte_path.name, 'num_documentos': 0, 'tokens_totais': 0, 'tokens_unicos': set(), 'tamanho_bytes': tamanho_bytes}

        with open(parte_path, 'r', encoding='utf-8') as f:
            documentos = [linha.strip() for linha in f if linha.strip()]

        textos_limpos = (limpar_texto(doc) for doc in documentos)
        docs_spacy = nlp_simples.pipe(textos_limpos, batch_size=2000, n_process=1)

        for doc in docs_spacy:
            tokens_min = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

            if tokens_min:
                stats_min_parte['num_documentos'] += 1
                stats_min_parte['tokens_totais'] += len(tokens_min)
                stats_min_parte['tokens_unicos'].update(tokens_min)

                tokens_sw = [t for t in tokens_min if t not in STOP_WORDS]

                if tokens_sw:
                    stats_sw_parte['num_documentos'] += 1
                    stats_sw_parte['tokens_totais'] += len(tokens_sw)
                    stats_sw_parte['tokens_unicos'].update(tokens_sw)

        stats_min_partes.append(stats_min_parte)
        stats_sw_partes.append(stats_sw_parte)

        stats_min_gerais['num_documentos'] += stats_min_parte['num_documentos']
        stats_min_gerais['tokens_totais'] += stats_min_parte['tokens_totais']
        stats_min_gerais['tokens_unicos'].update(stats_min_parte['tokens_unicos'])
        stats_min_gerais['tamanho_bytes'] += stats_min_parte['tamanho_bytes']

        stats_sw_gerais['num_documentos'] += stats_sw_parte['num_documentos']
        stats_sw_gerais['tokens_totais'] += stats_sw_parte['tokens_totais']
        stats_sw_gerais['tokens_unicos'].update(stats_sw_parte['tokens_unicos'])
        stats_sw_gerais['tamanho_bytes'] += stats_sw_parte['tamanho_bytes']

    return stats_min_gerais, stats_min_partes, stats_sw_gerais, stats_sw_partes


def analisar_estrategia_completa(partes_tokenizadas):
    """Analisa os arquivos já tokenizados (estratégia completa)."""
    print("\nAnalisando com a estratégia: 'Completa (Lematizada)'...")
    stats_gerais = {'num_documentos': 0, 'tokens_totais': 0, 'tokens_unicos': set(), 'tamanho_bytes': 0}
    stats_por_parte = []
    
    for parte_path in tqdm(partes_tokenizadas, desc="Estratégia 'Completa'"):
        stats_parte = {'nome': parte_path.name, 'num_documentos': 0, 'tokens_totais': 0, 'tokens_unicos': set(), 'tamanho_bytes': parte_path.stat().st_size}
        
        with open(parte_path, 'r', encoding='utf-8') as f:
            for linha in f:
                tokens = linha.strip().split(';')
                if tokens and tokens != ['']:
                    stats_parte['num_documentos'] += 1
                    stats_parte['tokens_totais'] += len(tokens)
                    stats_parte['tokens_unicos'].update(tokens)
        
        stats_por_parte.append(stats_parte)

        stats_gerais['num_documentos'] += stats_parte['num_documentos']
        stats_gerais['tokens_totais'] += stats_parte['tokens_totais']
        stats_gerais['tokens_unicos'].update(stats_parte['tokens_unicos'])
        stats_gerais['tamanho_bytes'] += stats_parte['tamanho_bytes']
        
    return stats_gerais, stats_por_parte

def gerar_graficos(stats_map, output_dir):
    """Gera gráficos comparativos."""
    print(f"\nGerando gráficos em '{output_dir}'...")
    output_dir.mkdir(exist_ok=True)
    
    labels = list(stats_map.keys())
    
    def plot_bar_chart(filename, title, data_key, log_scale=False, normalize=False):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Usa 'num_tokens_unicos' para o gráfico de vocabulário
        if data_key == 'num_tokens_unicos':
            values = np.array([stats['num_tokens_unicos'] for stats in stats_map.values()], dtype=np.float64)
        else:
            values = np.array([stats[data_key] for stats in stats_map.values()], dtype=np.float64)

        ax.set_ylabel(title, fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)

        if normalize and values[0] > 0:
            max_val = values[0]  # Assumes 'Mínima' is the reference
            values = (values / max_val) * 100
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            log_scale = False # Log scale is not ideal for normalized percentages
        
        bars = ax.bar(labels, values, color=[STRATEGY_COLORS.get(label) for label in labels])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE, rotation=0)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

        for bar in bars:
            height = bar.get_height()
            label_format = f'{height:.1f}%' if normalize else f'{height:,.0f}'
            ax.annotate(label_format,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
        
        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

        plt.tight_layout()
        plt.savefig(output_dir / f"{filename}.png", dpi=300)
        plt.close()

    # Gráficos existentes (com escala de log)
    plot_bar_chart('comparativo_tokens_totais_log', 'Tokens Totais (Escala Log)', 'tokens_totais', log_scale=True)
    plot_bar_chart('comparativo_tokens_unicos_log', 'Tokens Únicos (Escala Log)', 'num_tokens_unicos', log_scale=True)
    
    # Novo gráfico normalizado para tokens totais
    plot_bar_chart('comparativo_tokens_totais_normalizado', "Tokens Totais (em % de 'Mínima')", 'tokens_totais', normalize=True)
    
    print("✓ Gráficos gerais gerados com sucesso.")


def main():
    """Função principal."""
    start_time = time.time()
    
    print("=" * 70)
    print("ANÁLISE COMPARATIVA DE ESTRATÉGIAS DE TOKENIZAÇÃO")
    print("=" * 70)
    
    corpus_dir = Path("corpus")
    tokenizado_dir = corpus_dir / "tokenizado"
    output_dir = Path("analise_resultados")
    
    partes_originais = sorted(list(corpus_dir.glob("parte_*.txt")))
    partes_tokenizadas = sorted(list(tokenizado_dir.glob("parte_*.txt")))
    
    if not partes_originais or not partes_tokenizadas:
        print(f"\nErro: Corpus não encontrado! Verifique '{corpus_dir}' e '{tokenizado_dir}'.\n")
        return

    stats_minima_gerais, stats_minima_partes, stats_minima_sw_gerais, stats_minima_sw_partes = analisar_estrategias_basicas(partes_originais)
    stats_completa_gerais, stats_completa_partes = analisar_estrategia_completa(partes_tokenizadas)

    all_general_stats = {
        "Mínima": stats_minima_gerais,
        "Mínima+SW": stats_minima_sw_gerais,
        "Completa": stats_completa_gerais
    }
    for stats in all_general_stats.values():
        stats['num_tokens_unicos'] = len(stats['tokens_unicos'])

    print("\n" + "=" * 70)
    print("RESULTADOS GERAIS DA ANÁLISE")
    print("=" * 70)
    
    labels = list(all_general_stats.keys())
    header = f"\n{'Métrica':<25} | {labels[0]:>15} | {labels[1]:>15} | {labels[2]:>15}"
    print(header)
    print("-" * (28 + 18 * len(labels)))
    
    def print_row(metric_name, key, stats_dict, is_float=False):
        values = []
        for label in labels:
            stat_value = stats_dict[label].get(key, 0)
            if is_float:
                values.append(f"{stat_value:>15.2f}")
            else:
                values.append(f"{stat_value:>15,d}")
        print(f" {metric_name:<24} | {' | '.join(values)}")

    #print_row('Documentos', 'num_documentos', all_general_stats)
    print_row('Tokens Totais', 'tokens_totais', all_general_stats)
    print_row('Tokens Únicos', 'num_tokens_unicos', all_general_stats)

    avg_values = [f"{(s['tokens_totais'] / s['num_documentos'] if s['num_documentos'] > 0 else 0):>15.2f}" for s in all_general_stats.values()]
    print(f" {'Tokens/Documento (Média)':<24} | {' | '.join(avg_values)}")

    for stats in all_general_stats.values():
        stats['tamanho_mb'] = stats['tamanho_bytes'] / (1024*1024)
    print_row('Tamanho em Disco (MB)', 'tamanho_mb', all_general_stats, is_float=True)
    
    gerar_graficos(all_general_stats, output_dir)
    
    print("\n" + "=" * 70)
    print("RESULTADOS DETALHADOS POR PARTE")
    print("=" * 70)

    all_parts_data = OrderedDict()
    for parte in stats_minima_partes:
        all_parts_data[parte['nome']] = {"Mínima": parte}
    for parte in stats_minima_sw_partes:
        all_parts_data[parte['nome']]["Mínima+SW"] = parte
    for parte in stats_completa_partes:
        all_parts_data[parte['nome']]["Completa"] = parte
        
    for part_name, stats_map in all_parts_data.items():
        print(f"\n--- {part_name} ---")
        
        for stats in stats_map.values():
            stats['num_tokens_unicos'] = len(stats['tokens_unicos'])
        
        print_row('Documentos', 'num_documentos', stats_map)
        print_row('Tokens Totais', 'tokens_totais', stats_map)
        print_row('Tokens Únicos', 'num_tokens_unicos', stats_map)

        avg_values_part = [f"{(s['tokens_totais'] / s['num_documentos'] if s['num_documentos'] > 0 else 0):>15.2f}" for s in stats_map.values()]
        print(f" {'Tokens/Documento (Média)':<24} | {' | '.join(avg_values_part)}")

        for stats in stats_map.values():
            stats['tamanho_mb'] = stats['tamanho_bytes'] / (1024*1024)
        print_row('Tamanho em Disco (MB)', 'tamanho_mb', stats_map, is_float=True)

    total_time = time.time() - start_time
    print(f"\n\nAnálise concluída em {total_time:.2f} segundos.")
    print("=" * 70)

if __name__ == "__main__":
    main()
