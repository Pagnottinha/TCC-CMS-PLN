#!/usr/bin/env python3
"""
Comparação de ranking top-k PPMI entre estruturas de contagem.

Modos de execução:
- Padrão: Executa benchmark completo
- --plot-from-csv: Gera gráficos a partir de CSV existente
- --min-freq: Define limiar mínimo de frequência para pares
"""

import os
import time
import math
import heapq
import warnings
import argparse
import csv # Importar csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.stats import spearmanr

from factory_estruturas import carregar_estrutura, estimar_estrutura

import json

load_dotenv()
SEED = int(os.getenv('SEED', 42))
JANELA = int(os.getenv('JANELA', 3))
K_VALUES = [10, 100, 1000, 10000, 100000]

CORPUS_DIR = Path('corpus/tokenizado')
BENCHMARK_DIR = Path('resultados_benchmark')
OUTPUT_DIR = Path('analise_resultados/comparacao_ppmi')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORES = json.loads(os.getenv('CORES', '{}'))
LABEL_FONTSIZE = int(os.getenv('LABEL_FONTSIZE', 16))
LEGEND_FONTSIZE = int(os.getenv('LEGEND_FONTSIZE', 16))
TICK_LABEL_FONTSIZE = int(os.getenv('TICK_LABEL_FONTSIZE', 14))
LINEWIDTH = float(os.getenv('LINEWIDTH', 2.5))
FONTWEIGHT = os.getenv('FONTWEIGHT', 'bold')


def carregar_contagens_reais_part(parte_num: int) -> Tuple[Counter, Counter, int]:
    """Carrega contagens de palavras e pares (ground truth) de arquivos CSV."""
    word_counts = Counter()
    pair_counts = Counter()
    
    pasta_counter = BENCHMARK_DIR / 'Counter'
    caminho_palavras = pasta_counter / f'palavras_parte{parte_num:02d}.csv'
    caminho_pares = pasta_counter / f'pares_parte{parte_num:02d}.csv'

    if not caminho_palavras.exists():
        raise FileNotFoundError(
            f"Arquivo de contagens de palavras para a parte {parte_num:02d} não encontrado em "
            f"'{caminho_palavras}'. Execute o script 'benchmark_estruturas.py' primeiro."
        )
    if not caminho_pares.exists():
        raise FileNotFoundError(
            f"Arquivo de contagens de pares para a parte {parte_num:02d} não encontrado em "
            f"'{caminho_pares}'. Execute o script 'benchmark_estruturas.py' primeiro."
        )

    with open(caminho_palavras, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            word_counts[row[0]] = int(row[1])

    with open(caminho_pares, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pair_counts[row[0]] = int(row[1])

    total_words = sum(word_counts.values())
    
    return word_counts, pair_counts, total_words


def calcular_ppmi(
    word_counts: Dict[str, int], 
    pair_counts: Dict[str, int], 
    total_words: int
) -> Dict[str, float]:
    """Calcula scores PPMI para todos os pares de palavras."""
    if not total_words:
        return {}
        
    ppmi_scores = {}
    
    for pair, pair_count in pair_counts.items():
        tokens = pair.split(' ')
        if len(tokens) != 2: continue
        
        token1, token2 = tokens
        
        word_count1 = word_counts.get(token1, 0)
        word_count2 = word_counts.get(token2, 0)

        if word_count1 == 0 or word_count2 == 0:
            continue

        p_pair = pair_count / total_words
        p_word1 = word_count1 / total_words
        p_word2 = word_count2 / total_words
        
        if p_pair == 0 or p_word1 == 0 or p_word2 == 0:
            pmi = -float('inf')
        else:
            pmi = math.log2(p_pair / (p_word1 * p_word2))
        
        ppmi = max(0, pmi)
        
        if ppmi > 0:
            ppmi_scores[pair] = ppmi
            
    return ppmi_scores

def obter_top_k(ppmi_scores: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    """Retorna os k pares com maiores scores de PPMI."""
    return heapq.nlargest(k, ppmi_scores.items(), key=lambda item: item[1])

def gerar_graficos(df_parts: pd.DataFrame, file_suffix: str = ""):
    """Gera e salva os gráficos de métricas."""
    print("\nGerando gráficos...")
    for metrica in ['recall', 'spearman_corr']:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.lineplot(
            data=df_parts, x='k', y=metrica, hue='estrutura',
            marker='o', palette=CORES, errorbar=None, ax=ax, linewidth=LINEWIDTH
        )
        
        ax.set_xscale('log')
        
        title_map = {
            'recall': 'Recall',
            'spearman_corr': 'Correlação de Spearman'
        }
        metrica_titulo = title_map.get(metrica, metrica.capitalize())
        

        ax.set_xlabel('k (em escala log)', fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
        ax.set_ylabel(f'{metrica_titulo} Média', fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
        
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        ax.legend(title='Estrutura', fontsize=LEGEND_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        
        plt.tight_layout()
        
        fig_path = OUTPUT_DIR / f'grafico_{metrica}_ppmi_medio{file_suffix}.png'
        plt.savefig(fig_path, dpi=300)
        print(f"✓ Gráfico salvo em: {fig_path}")
        plt.close(fig)

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Compara rankings PPMI e opcionalmente gera gráficos a partir de CSV.")
    parser.add_argument(
        '--plot-from-csv',
        action='store_true',
        help="Apenas gera os gráficos a partir do arquivo CSV 'comparacao_ranking_ppmi_por_parte.csv' existente, pulando o cálculo."
    )
    parser.add_argument(
        '--min-freq',
        type=int,
        default=1,
        help="Limiar mínimo de frequência para que um par de palavras seja considerado (default: 1)."
    )
    parser.add_argument(
        '--show-top-10',
        action='store_true',
        help="Exibe o ranking top-10 de pares PPMI para cada parte do corpus (ground truth)."
    )
    parser.add_argument(
        '--use-exact-word-freq',
        action='store_true',
        help="Utiliza as frequências exatas de palavras (ground truth) nos cálculos de PPMI para as estruturas probabilísticas, em vez de estimar as frequências de palavras."
    )
    args = parser.parse_args()

    file_suffix = f"_minfreq_{args.min_freq}" if args.min_freq > 1 else ""
    csv_parts_base_name = f"comparacao_ranking_ppmi_por_parte{file_suffix}.csv"
    csv_parts_path = OUTPUT_DIR / csv_parts_base_name

    if args.plot_from_csv:
        print("Modo de plotagem a partir de CSV ativado.")
        if not csv_parts_path.exists():
            print(f"Erro: Arquivo de resultados '{csv_parts_path}' não encontrado. Execute o script sem a flag para gerá-lo.")
            return
        
        print(f"Carregando dados de '{csv_parts_path}'...")
        df_parts = pd.read_csv(csv_parts_path)
        gerar_graficos(df_parts, file_suffix)
        print("\nProcesso de plotagem concluído com sucesso!")
        return

    print("=" * 80)
    print("Comparação de Ranking Top-k PPMI entre Estruturas (por Parte)")
    print(f"Limiar de frequência mínima para pares: {args.min_freq}")
    print("=" * 80)

    all_results_parts = []
    nomes_estruturas = [nome for nome in os.listdir(BENCHMARK_DIR) if (BENCHMARK_DIR / nome).is_dir() and nome != 'Counter']
    
    arquivos_corpus = sorted(list(CORPUS_DIR.glob("parte*.txt")))
    max_k = max(K_VALUES)

    if not arquivos_corpus:
        print(f"Nenhum arquivo de corpus encontrado em '{CORPUS_DIR}'. Saindo.")
        return

    for arquivo_path in tqdm(arquivos_corpus, desc="Processando partes do corpus"):
        parte_num_str = ''.join(filter(str.isdigit, arquivo_path.stem))
        parte_num = int(parte_num_str)

        true_word_counts, true_pair_counts, total_words = carregar_contagens_reais_part(parte_num)
        
        filtered_true_pair_counts = {p: c for p, c in true_pair_counts.items() if c >= args.min_freq}


        true_ppmi_scores = calcular_ppmi(true_word_counts, filtered_true_pair_counts, total_words)
        
        if args.show_top_10:
            print(f"\n--- Top 10 PPMI - {arquivo_path.name} ---")
            top_10_pairs = obter_top_k(true_ppmi_scores, 10)
            for i, (pair, score) in enumerate(top_10_pairs):
                print(f"{i+1:2d}. '{pair}' (Score: {score:.4f})")
            print("-" * (30 + len(arquivo_path.name)))
            
        true_top_max_k = obter_top_k(true_ppmi_scores, max_k)
        ground_truth_sets: Dict[int, Set[str]] = {
            k: {pair for pair, score in true_top_max_k[:k]}
            for k in K_VALUES
        }

        for nome_estrutura in tqdm(nomes_estruturas, desc=f"Benchmarking {arquivo_path.name}", leave=False):
            caminho_estrutura = BENCHMARK_DIR / nome_estrutura / f"parte{parte_num_str}"

            try:
                estrutura = carregar_estrutura(nome_estrutura, str(caminho_estrutura))
            except FileNotFoundError:
                print(f"Aviso: Arquivo de estrutura não encontrado para {nome_estrutura} na parte {parte_num_str}. Pulando.")
                continue

            estimated_word_counts = {}
            if args.use_exact_word_freq:
                estimated_word_counts = true_word_counts
            else:
                estimated_word_counts = {word: estimar_estrutura(estrutura, word) for word in true_word_counts.keys()}
            estimated_pair_counts = {pair: estimar_estrutura(estrutura, pair) for pair in true_pair_counts.keys()}

            filtered_estimated_pair_counts = {p: c for p, c in estimated_pair_counts.items() if c >= args.min_freq}

            estimated_ppmi_scores = calcular_ppmi(
                estimated_word_counts, 
                filtered_estimated_pair_counts,
                total_words
            )
            
            estimated_top_max_k = obter_top_k(estimated_ppmi_scores, max_k)

            for k in K_VALUES:
                true_set = ground_truth_sets.get(k, set())
                if not true_set: continue

                estimated_top_k = {pair for pair, score in estimated_top_max_k[:k]}
                
                intersection_size = len(true_set.intersection(estimated_top_k))
                
                recall = intersection_size / len(true_set) if true_set else 0

                all_pairs_in_top_k = true_set.union(estimated_top_k)
                true_values = [true_ppmi_scores.get(pair, 0.0) for pair in all_pairs_in_top_k]
                estimated_values = [estimated_ppmi_scores.get(pair, 0.0) for pair in all_pairs_in_top_k]
                
                spearman_correlation = 0.0
                if np.std(true_values) > 0 and np.std(estimated_values) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        spearman_correlation, _ = spearmanr(true_values, estimated_values)
                        if np.isnan(spearman_correlation): spearman_correlation = 0.0
                
                all_results_parts.append({
                    'parte': arquivo_path.name, 'estrutura': nome_estrutura, 'k': k,
                    'recall': recall,
                    'spearman_corr': spearman_correlation
                })

    print("\n" + "=" * 80)
    print("Agregando e salvando resultados...")
    if not all_results_parts:
        print("Nenhum resultado gerado.")
        return

    df_parts = pd.DataFrame(all_results_parts)
    
    df_agg = df_parts.groupby(['estrutura', 'k']).agg(
        recall_media=('recall', 'mean'),
        spearman_corr_media=('spearman_corr', 'mean')
    ).reset_index()

    df_agg.to_csv(OUTPUT_DIR / f'comparacao_ranking_ppmi_agregado{file_suffix}.csv', index=False)
    df_parts.to_csv(csv_parts_path, index=False)
    print(f"✓ Resultados salvos em: {OUTPUT_DIR}")

    gerar_graficos(df_parts, file_suffix)

    print("\nProcesso concluído com sucesso!")

if __name__ == '__main__':
    main()