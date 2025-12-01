#!/usr/bin/env python3
"""Gera gráficos de MRE em função da frequência dos tokens."""

import json
import csv
import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from factory_estruturas import (
    obter_classe_estrutura,
    estimar_estrutura,
    CONFIGS_PADRAO as CONFIGS
)

load_dotenv()

BENCHMARK_DIR = Path('resultados_benchmark')
ANALISE_DIR = Path('analise_resultados')
ANALISE_DIR.mkdir(exist_ok=True)

ESTRUTURAS_A_ANALISAR = list(CONFIGS.keys())
ESTRUTURAS_SUPER_SUB = ['CML8S-CU', 'CML8HS-CU']

CORES = json.loads(os.getenv('CORES', '{}'))
LABEL_FONTSIZE = int(os.getenv('LABEL_FONTSIZE', 16))
LEGEND_FONTSIZE = int(os.getenv('LEGEND_FONTSIZE', 16))
TICK_LABEL_FONTSIZE = int(os.getenv('TICK_LABEL_FONTSIZE', 14))
LINEWIDTH = float(os.getenv('LINEWIDTH', 2.5))
FONTWEIGHT = os.getenv('FONTWEIGHT', 'bold')

SUPER_SUB_MARKERS = {
    'CML8S-CU': {'over': 'o', 'under': 's'}, # circle and square
    'CML8HS-CU': {'over': '^', 'under': 'v'}, # original triangle up/down
}
SUPER_SUB_LINESTYLES = {
    'CML8S-CU': {'over': '-', 'under': ':'}, # solid and dotted
    'CML8HS-CU': {'over': '-', 'under': '--'}, # original solid and dashed
}

SUPER_SUB_CORES = {
    "CML8S-CU": {"over": "#1f77b4", "under": "#880fb8ff"},
    "CML8HS-CU": {"over": "#2ca02c", "under": "#8fbd13ff"}
}


def carregar_e_processar_dados_por_parte():
    """Carrega dados e calcula MRE por frequência."""
    print("Carregando e processando dados (calculando MRE por parte)...")
    
    dados_agregados = {
        'geral': {
            nome: defaultdict(list)
            for nome in ESTRUTURAS_A_ANALISAR
        },
        'super_sub': {
            nome: {
                'over': defaultdict(list),
                'under': defaultdict(list)
            } for nome in ESTRUTURAS_SUPER_SUB
        }
    }

    pasta_counter = BENCHMARK_DIR / 'Counter'
    num_partes = len(list(pasta_counter.glob('palavras_parte*.csv')))
    
    if num_partes == 0:
        print("Nenhum resultado de benchmark encontrado (.csv).")
        return None

    for i in tqdm(range(1, num_partes + 1), desc="Processando partes", ncols=70):
        estruturas = {}
        for nome_estrutura in ESTRUTURAS_A_ANALISAR:
            caminho_estrutura = BENCHMARK_DIR / nome_estrutura / f"parte{i:02d}"
            try:
                classe_estrutura = obter_classe_estrutura(nome_estrutura)
                estruturas[nome_estrutura] = classe_estrutura.carregar(str(caminho_estrutura))
            except FileNotFoundError:
                print(f"\nAlerta: Arquivo de dados para a estrutura '{nome_estrutura}' "
                      f"na parte {i} não encontrado em '{caminho_estrutura}'. "
                      "A estrutura será ignorada nesta parte.")
                continue
        
        if not estruturas:
            print(f"\nAlerta: Nenhuma estrutura encontrada para a parte {i}. Pulando.")
            continue

        # Dicionários para acumular erros apenas para a parte ATUAL
        erros_na_parte = {
            nome: defaultdict(lambda: {'sum': 0.0, 'count': 0})
            for nome in ESTRUTURAS_A_ANALISAR
        }
        erros_super_sub_na_parte = {
            nome: defaultdict(lambda: {
                'over_sum': 0.0, 'over_count': 0,
                'under_sum': 0.0, 'under_count': 0
            }) for nome in ESTRUTURAS_SUPER_SUB
        }

        for tipo_contador in ['palavras', 'pares']:
            caminho_contador = pasta_counter / f'{tipo_contador}_parte{i:02d}.csv'
            if not caminho_contador.exists():
                continue

            with open(caminho_contador, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    token, freq_real_str = row
                    freq_real = int(freq_real_str)

                    if freq_real > 0:
                        for nome_estrutura, estrutura in estruturas.items():
                            freq_estimada = estimar_estrutura(estrutura, token)
                            erro_bruto = freq_estimada - freq_real
                            erro_relativo = abs(erro_bruto) / freq_real
                            
                            # Acumula erros para MRE geral DENTRO da parte
                            agg_geral = erros_na_parte[nome_estrutura][freq_real]
                            agg_geral['sum'] += erro_relativo
                            agg_geral['count'] += 1

                            # Acumula erros para super/subestimação DENTRO da parte
                            if nome_estrutura in ESTRUTURAS_SUPER_SUB:
                                agg_super_sub = erros_super_sub_na_parte[nome_estrutura][freq_real]
                                if erro_bruto > 0:
                                    agg_super_sub['over_sum'] += erro_relativo
                                    agg_super_sub['over_count'] += 1
                                elif erro_bruto < 0:
                                    agg_super_sub['under_sum'] += erro_relativo
                                    agg_super_sub['under_count'] += 1
        
        # Após processar a parte, calcula MRE por freq para ESTA parte e armazena
        for nome_estrutura, freqs in erros_na_parte.items():
            for freq, agg in freqs.items():
                if agg['count'] > 0:
                    mre_da_freq_na_parte = agg['sum'] / agg['count']
                    dados_agregados['geral'][nome_estrutura][freq].append(mre_da_freq_na_parte)

        for nome_estrutura, freqs in erros_super_sub_na_parte.items():
            for freq, agg in freqs.items():
                if agg['over_count'] > 0:
                    mre_over = agg['over_sum'] / agg['over_count']
                    dados_agregados['super_sub'][nome_estrutura]['over'][freq].append(mre_over)
                if agg['under_count'] > 0:
                    mre_under = agg['under_sum'] / agg['under_count']
                    dados_agregados['super_sub'][nome_estrutura]['under'][freq].append(mre_under)

    return dados_agregados


def calcular_mre_medio(mre_por_freq_agregado):
    """Calcula a média dos MREs por frequência."""
    print("Calculando a média dos MREs por frequência...")
    dados_grafico = {nome: {'frequencias': [], 'mre_medio': []} for nome in ESTRUTURAS_A_ANALISAR}

    for nome_estrutura, freqs in mre_por_freq_agregado.items():
        for freq, lista_mres in sorted(freqs.items()):
            if freq > 0 and lista_mres:
                dados_grafico[nome_estrutura]['frequencias'].append(freq)
                dados_grafico[nome_estrutura]['mre_medio'].append(np.mean(lista_mres))
                
    return dados_grafico


def gerar_graficos(dados_grafico, freq_threshold=100):
    """Gera os gráficos de MRE vs. Frequência."""
    print("Gerando gráficos MRE vs Frequência...")
    
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for nome_estrutura, dados in dados_grafico.items():
        dados_baixa_freq = [(f, mre) for f, mre in zip(dados['frequencias'], dados['mre_medio']) if f <= freq_threshold]
        if dados_baixa_freq:
            frequencias, mre_medio = zip(*dados_baixa_freq)
            ax1.plot(frequencias, mre_medio, marker='o', linestyle='-', markersize=4, alpha=0.8, label=nome_estrutura, color=CORES.get(nome_estrutura), linewidth=LINEWIDTH)
    
    ax1.set_xlabel(
        'Frequência do Token (Escala Linear)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax1.set_ylabel(
        'Erro Relativo Médio (MRE) (Escala Linear)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax1.legend(title='Estruturas', fontsize=LEGEND_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.tight_layout()
    caminho_salvar_baixa = ANALISE_DIR / 'mre_vs_frequencia_baixas.png'
    plt.savefig(caminho_salvar_baixa, dpi=300)
    print(f"✓ Gráfico de frequências baixas salvo em: {caminho_salvar_baixa}")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for nome_estrutura, dados in dados_grafico.items():
        if dados['frequencias']:
            ax2.plot(dados['frequencias'], dados['mre_medio'], marker='o', linestyle='-', markersize=2, alpha=0.8, label=nome_estrutura, color=CORES.get(nome_estrutura), linewidth=LINEWIDTH)
    
    ax2.set_xlabel(
        'Frequência do Token (Escala Log)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax2.set_ylabel(
        'Erro Relativo Médio (MRE) (Escala Log)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax2.legend(title='Estruturas', fontsize=LEGEND_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.tight_layout()
    caminho_salvar_geral = ANALISE_DIR / 'mre_vs_frequencia_geral_log.png'
    plt.savefig(caminho_salvar_geral, dpi=300)
    print(f"✓ Gráfico geral salvo em: {caminho_salvar_geral}")
    plt.close(fig2)


def calcular_mre_super_sub(erros_super_sub_agregado):
    """Calcula MRE de super e subestimação por frequência."""
    print("Calculando MRE médio de super/subestimação...")
    dados_grafico = {
        nome: {'frequencias': [], 'mre_over': [], 'mre_under': []}
        for nome in ESTRUTURAS_SUPER_SUB
    }
    for nome_estrutura, dados in erros_super_sub_agregado.items():
        # Agrega todas as frequências únicas de 'over' e 'under'
        todas_as_freqs = sorted(set(dados['over'].keys()) | set(dados['under'].keys()))
        
        for freq in todas_as_freqs:
            if freq > 0:
                dados_grafico[nome_estrutura]['frequencias'].append(freq)
                
                lista_mre_over = dados['over'].get(freq, [])
                mre_over = np.mean(lista_mre_over) if lista_mre_over else 0
                dados_grafico[nome_estrutura]['mre_over'].append(mre_over)
                
                lista_mre_under = dados['under'].get(freq, [])
                mre_under = np.mean(lista_mre_under) if lista_mre_under else 0
                dados_grafico[nome_estrutura]['mre_under'].append(mre_under)
                
    return dados_grafico


def gerar_graficos_super_sub(dados_grafico, freq_threshold=100):
    """Gera gráficos de MRE para super/subestimação."""
    print("Gerando gráficos de super/subestimação...")

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for nome_estrutura, dados in dados_grafico.items():
        indices = [i for i, f in enumerate(dados['frequencias']) if f <= freq_threshold]
        if not indices: continue
        
        frequencias = [dados['frequencias'][i] for i in indices]
        mre_over = [dados['mre_over'][i] for i in indices]
        mre_under = [dados['mre_under'][i] for i in indices]
        
        cores = SUPER_SUB_CORES.get(nome_estrutura, {})
        cor_over = cores.get('over')
        cor_under = cores.get('under')

        ax1.plot(
            frequencias, mre_over,
            marker=SUPER_SUB_MARKERS[nome_estrutura]['over'],
            linestyle=SUPER_SUB_LINESTYLES[nome_estrutura]['over'],
            markersize=5,
            alpha=0.9,
            label=f'{nome_estrutura} (Super)',
            color=cor_over,
            linewidth=LINEWIDTH
        )
        ax1.plot(
            frequencias, mre_under,
            marker=SUPER_SUB_MARKERS[nome_estrutura]['under'],
            linestyle=SUPER_SUB_LINESTYLES[nome_estrutura]['under'],
            markersize=5,
            alpha=0.9,
            label=f'{nome_estrutura} (Sub)',
            color=cor_under,
            linewidth=LINEWIDTH
        )
    
    ax1.set_xlabel(
        'Frequência do Token (Escala Linear)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax1.set_ylabel(
        'Erro Relativo Médio (MRE) (Escala Linear)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax1.legend(title='Estruturas', fontsize=LEGEND_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.tight_layout()
    caminho_salvar_baixa = ANALISE_DIR / 'mre_super_sub_frequencia_baixas.png'
    plt.savefig(caminho_salvar_baixa, dpi=300)
    print(f"✓ Gráfico salvo em: {caminho_salvar_baixa}")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for nome_estrutura, dados in dados_grafico.items():
        if dados['frequencias']:
            cores = SUPER_SUB_CORES.get(nome_estrutura, {})
            cor_over = cores.get('over')
            cor_under = cores.get('under')

            ax2.plot(
                dados['frequencias'], dados['mre_over'],
                marker=SUPER_SUB_MARKERS[nome_estrutura]['over'],
                markersize=5,
                linestyle=SUPER_SUB_LINESTYLES[nome_estrutura]['over'],
                alpha=0.9,
                label=f'{nome_estrutura} (Super)',
                color=cor_over,
                linewidth=LINEWIDTH
            )
            ax2.plot(
                dados['frequencias'], dados['mre_under'],
                marker=SUPER_SUB_MARKERS[nome_estrutura]['under'],
                markersize=5,
                linestyle=SUPER_SUB_LINESTYLES[nome_estrutura]['under'],
                alpha=0.9,
                label=f'{nome_estrutura} (Sub)',
                color=cor_under,
                linewidth=LINEWIDTH
            )
    
    ax2.set_xlabel(
        'Frequência do Token (Escala Log)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax2.set_ylabel(
        'Erro Relativo Médio (MRE) (Escala Log)',
        fontsize=LABEL_FONTSIZE,
        fontweight=FONTWEIGHT
    )
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax2.legend(title='Estruturas', fontsize=LEGEND_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.tight_layout()
    caminho_salvar_geral = ANALISE_DIR / 'mre_super_sub_frequencia_geral_log.png'
    plt.savefig(caminho_salvar_geral, dpi=300)
    print(f"✓ Gráfico geral salvo em: {caminho_salvar_geral}")
    plt.close(fig2)


def main():
    """Função principal."""
    dados_agregados = carregar_e_processar_dados_por_parte()
    
    if dados_agregados:
        dados_gerais = calcular_mre_medio(dados_agregados['geral'])
        gerar_graficos(dados_gerais)
        
        dados_super_sub = calcular_mre_super_sub(dados_agregados['super_sub'])
        gerar_graficos_super_sub(dados_super_sub)
    else:
        print("Finalizado sem gerar gráfico.")


if __name__ == '__main__':
    main()
