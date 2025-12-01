"""Análise de distribuição Zipf de palavras e pares no corpus."""
from pathlib import Path
from collections import Counter
import csv
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI para renderização mais rápida
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv

load_dotenv()

plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

CORES = json.loads(os.getenv('CORES', '{}'))
LABEL_FONTSIZE = int(os.getenv('LABEL_FONTSIZE', 16))
LEGEND_FONTSIZE = int(os.getenv('LEGEND_FONTSIZE', 16)) # Corrected default to 16
TICK_LABEL_FONTSIZE = int(os.getenv('TICK_LABEL_FONTSIZE', 14))
LINEWIDTH = float(os.getenv('LINEWIDTH', 2.5))
FONTWEIGHT = os.getenv('FONTWEIGHT', 'bold')

GLOBAL_TOTAL_COLOR = 'red'
IDEAL_LINE_COLOR = 'black'
IDEAL_LINE_STYLE = '--'

def amostrar_logspace(x, y, max_pontos=10000):
    """Amostra pontos em escala logarítmica."""
    n = len(x)
    if n <= max_pontos:
        return x, y
    indices = np.logspace(0, np.log10(n - 1), max_pontos, dtype=int)
    indices = np.insert(indices, 0, 0)
    indices = np.unique(indices)
    return x[indices], y[indices]

def plotar_zipf_rank_frequency(dados_partes, dados_global, arquivo_saida, titulo):
    """Plota o gráfico de Zipf (Rank vs. Frequência)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    print(f"\n  Gerando gráfico de Zipf (Rank vs. Frequência) para '{titulo}'...")

    n_partes = len(dados_partes)
    cores_partes = plt.cm.tab20(np.linspace(0, 1, n_partes))
    estilos_linha = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    marcadores = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']

    for idx, dados in enumerate(dados_partes):
        ranks = np.arange(1, len(dados['frequencias']) + 1)
        freqs = dados['frequencias']
        ranks_plot, freqs_plot = amostrar_logspace(ranks, freqs, max_pontos=50000)
        ax.loglog(ranks_plot, freqs_plot, linestyle=estilos_linha[idx % len(estilos_linha)], 
                  alpha=1.0, linewidth=1.2, color=cores_partes[idx], 
                  marker=marcadores[idx % len(marcadores)], markevery=0.2, 
                  markersize=4, label=dados['nome'], rasterized=True)

    freqs_global = dados_global['frequencias']
    ranks_global = np.arange(1, len(freqs_global) + 1)
    ranks_plot, freqs_plot = amostrar_logspace(ranks_global, freqs_global, max_pontos=200000)
    ax.loglog(ranks_plot, freqs_plot, color=GLOBAL_TOTAL_COLOR, linestyle='-', 
               linewidth=LINEWIDTH, label=dados_global['nome'], zorder=100, rasterized=True)

    freq_1 = freqs_global[0] if len(freqs_global) > 0 else 1
    zipf_ideal = freq_1 / ranks_plot
    ax.loglog(ranks_plot, zipf_ideal, color=IDEAL_LINE_COLOR, linestyle=IDEAL_LINE_STYLE, 
               linewidth=1.5, alpha=0.7, label='Lei de Zipf (α=1)', zorder=90, rasterized=True)

    ax.set_xlabel('Ranking (log)', fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    ax.set_ylabel('Frequência (log)', fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE, ncol=2, framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Gráfico salvo em: {arquivo_saida}")

def plotar_ccdf(dados_partes, dados_global, arquivo_saida, titulo):
    """Plota o gráfico CCDF."""
    fig, ax = plt.subplots(figsize=(12, 8))
    print(f"  Gerando gráfico CCDF para '{titulo}'...")

    def calcular_ccdf_otimizado(frequencias):
        if len(frequencias) == 0: return np.array([]), np.array([])
        valores_unicos, counts = np.unique(frequencias, return_counts=True)
        ordem = np.argsort(valores_unicos)[::-1]
        valores_unicos = valores_unicos[ordem]
        counts = counts[ordem]
        ccdf = np.cumsum(counts) / len(frequencias)
        return valores_unicos, ccdf

    n_partes = len(dados_partes)
    cores_partes = plt.cm.tab20(np.linspace(0, 1, n_partes))
    estilos_linha = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    marcadores = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']

    for idx, dados in enumerate(dados_partes):
        valores, ccdf = calcular_ccdf_otimizado(dados['frequencias'])
        valores_plot, ccdf_plot = amostrar_logspace(valores, ccdf, max_pontos=50000)
        if len(valores_plot) > 0:
            ax.loglog(valores_plot, ccdf_plot, linestyle=estilos_linha[idx % len(estilos_linha)], alpha=1.0, 
                      linewidth=1.2, color=cores_partes[idx], marker=marcadores[idx % len(marcadores)], 
                      markevery=0.2, markersize=4, label=dados['nome'], rasterized=True)

    freqs_global = dados_global['frequencias']
    valores_global, ccdf_global = calcular_ccdf_otimizado(freqs_global)
    valores_plot, ccdf_plot = amostrar_logspace(valores_global, ccdf_global, max_pontos=200000)
    if len(valores_plot) > 0:
        ax.loglog(valores_plot, ccdf_plot, color=GLOBAL_TOTAL_COLOR, linestyle='-', 
                   linewidth=LINEWIDTH, label=dados_global['nome'], zorder=100, rasterized=True)
        alpha_ref = 2.0
        ccdf_ideal = (valores_plot) ** (1.0 - alpha_ref)
        ax.loglog(valores_plot, ccdf_ideal, color=IDEAL_LINE_COLOR, linestyle=IDEAL_LINE_STYLE, 
                   linewidth=1.5, alpha=0.7, label=f'CCDF ideal (α={alpha_ref:.1f})', zorder=90, rasterized=True)

    ax.set_xlabel('Frequência (log)', fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    ax.set_ylabel('P(X ≥ frequência) (log)', fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE, ncol=2, framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Gráfico salvo em: {arquivo_saida}")

def plotar_zipf(dados_partes, dados_global, arquivo_saida="corpus/grafico_zipf.png", titulo="N-gramas"):
    """Gera gráficos de Zipf e CCDF."""
    p = Path(arquivo_saida)
    arquivo_saida_rank = p.with_name(f"{p.stem}_rank_freq{p.suffix}")
    arquivo_saida_ccdf = p.with_name(f"{p.stem}_ccdf{p.suffix}")

    plotar_zipf_rank_frequency(dados_partes, dados_global, arquivo_saida_rank, titulo)
    plotar_ccdf(dados_partes, dados_global, arquivo_saida_ccdf, titulo)


def analisar_zipf():
    """Analisa distribuição de palavras e pares."""
    
    print("=" * 70)
    print("ANÁLISE DE DISTRIBUIÇÃO DE N-GRAMAS")
    print("=" * 70)
    
    # Diretório dos contadores salvos
    counter_dir = Path("resultados_benchmark/Counter")
    
    if not counter_dir.exists():
        print("\n❌ Erro: Diretório resultados_benchmark/Counter não encontrado!")
        print("   Execute primeiro o benchmark_estruturas.py")
        return False
    
    # Buscar arquivos de palavras e pares
    arquivos_palavras = sorted(list(counter_dir.glob("palavras_parte*.csv")))
    arquivos_pares = sorted(list(counter_dir.glob("pares_parte*.csv")))
    
    if not arquivos_palavras or not arquivos_pares:
        print("\n❌ Erro: Arquivos de contadores não encontrados no formato .csv!")
        print("   Execute primeiro o benchmark_estruturas.py para gerar os arquivos corretos.")
        return False
    
    print(f"\n✓ Encontrados {len(arquivos_palavras)} arquivos de palavras")
    print(f"✓ Encontrados {len(arquivos_pares)} arquivos de pares")
    
    # Contadores globais
    palavras_global = Counter()
    pares_global = Counter()
    
    # Listas para armazenar dados para os gráficos (separados + combinados)
    dados_para_grafico_uni = []
    dados_para_grafico_bi = []
    dados_para_grafico_combinado = []
    
    # Processar cada parte
    print(f"\n{'=' * 70}")
    print(f"Carregando {len(arquivos_palavras)} partes...")
    print(f"{ '=' * 70}\n")
    
    for idx in range(len(arquivos_palavras)):
        arquivo_pal = arquivos_palavras[idx]
        arquivo_par = arquivos_pares[idx]
        
        print(f"[{idx+1:02d}/{len(arquivos_palavras)}] Carregando {arquivo_pal.stem}")
        
        palavras_parte = Counter()
        print(f"  Carregando palavras...", end=' ', flush=True)
        with open(arquivo_pal, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader) # Pular cabeçalho
            for row in reader:
                palavras_parte[row[0]] = int(row[1])
        print(f"✓ {len(palavras_parte):,} distintas")
        
        pares_parte = Counter()
        print(f"  Carregando pares...", end=' ', flush=True)
        with open(arquivo_par, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader) # Pular cabeçalho
            for row in reader:
                pares_parte[row[0]] = int(row[1])
        print(f"✓ {len(pares_parte):,} distintas")
        
        # Converter para arrays numpy
        n_uni = len(palavras_parte)
        n_bi = len(pares_parte)
        
        freqs_uni = np.fromiter((freq for _, freq in palavras_parte.most_common()), dtype=np.int64, count=n_uni)
        freqs_bi = np.fromiter((freq for _, freq in pares_parte.most_common()), dtype=np.int64, count=n_bi)
        
        # Combinar palavras e pares desta parte
        ngramas_combinados = Counter()
        ngramas_combinados.update(palavras_parte)
        ngramas_combinados.update(pares_parte)
        
        n_comb = len(ngramas_combinados)
        freqs_combinados = np.fromiter((freq for _, freq in ngramas_combinados.most_common()), dtype=np.int64, count=n_comb)
        
        # Adicionar aos dados para gráficos
        nome_parte = f"parte_{idx+1:02d}"
        dados_para_grafico_uni.append({
            'nome': nome_parte,
            'frequencias': freqs_uni
        })
        
        dados_para_grafico_bi.append({
            'nome': nome_parte,
            'frequencias': freqs_bi
        })
        
        dados_para_grafico_combinado.append({
            'nome': nome_parte,
            'frequencias': freqs_combinados
        })
        
        # Atualizar contadores globais
        palavras_global.update(palavras_parte)
        pares_global.update(pares_parte)
        
        print(f"  ✓ Total combinado: {len(ngramas_combinados):,}")
        print()
        
        # Liberar memória
        del palavras_parte
        del pares_parte
        del ngramas_combinados
        gc.collect()
    
    # Análise de distribuição
    print(f"{ '=' * 70}")
    print("ANÁLISE DE DISTRIBUIÇÃO (LEI DE ZIPF)")
    print(f"{ '=' * 70}")
    
    # Estatísticas finais
    print("\n  Calculando totais de ocorrências...")
    total_ocorrencias_uni = sum(palavras_global.values())
    total_ocorrencias_bi = sum(pares_global.values())
    print(f"  ✓ Palavras: {total_ocorrencias_uni:,} ocorrências")
    print(f"  ✓ Pares: {total_ocorrencias_bi:,} ocorrências")
    
    # Combinar palavras e pares globais
    print("\n  Combinando palavras e pares globais...")
    ngramas_global_combinados = Counter()
    ngramas_global_combinados.update(palavras_global)
    print("    ✓ Palavras adicionadas")
    ngramas_global_combinados.update(pares_global)
    print("    ✓ Pares adicionados")
    total_ocorrencias_combinadas = sum(ngramas_global_combinados.values())
    print(f"  ✓ Total combinado: {total_ocorrencias_combinadas:,} ocorrências")
    
    # Preparar dados globais para os gráficos (separados + combinados)
    # Usar numpy arrays para eficiência com conversão otimizada
    print("\n  Convertendo contadores globais para arrays numpy...")
    
    n_uni_global = len(palavras_global)
    n_bi_global = len(pares_global)
    n_comb_global = len(ngramas_global_combinados)
    
    print(f"    [1/3] Palavras: {n_uni_global:,} distintas")
    freqs_uni_global = np.fromiter((freq for _, freq in palavras_global.most_common()), dtype=np.int64, count=n_uni_global)
    print(f"    ✓ Convertido em {freqs_uni_global.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"    [2/3] Pares: {n_bi_global:,} distintas")
    freqs_bi_global = np.fromiter((freq for _, freq in pares_global.most_common()), dtype=np.int64, count=n_bi_global)
    print(f"    ✓ Convertido em {freqs_bi_global.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"    [3/3] Combinados: {n_comb_global:,} distintas")
    freqs_combinados_global = np.fromiter((freq for _, freq in ngramas_global_combinados.most_common()), dtype=np.int64, count=n_comb_global)
    print(f"    ✓ Convertido em {freqs_combinados_global.nbytes / 1024 / 1024:.2f} MB")
    
    print("  ✓ Conversão concluída")
    
    dados_global_uni = {
        'nome': 'TOTAL GLOBAL',
        'frequencias': freqs_uni_global
    }
    
    dados_global_bi = {
        'nome': 'TOTAL GLOBAL',
        'frequencias': freqs_bi_global
    }
    
    dados_global_combinado = {
        'nome': 'TOTAL GLOBAL',
        'frequencias': freqs_combinados_global
    }
    
    # Gerar gráficos de Zipf (3 gráficos: combinado + separados)
    print(f"\n{'=' * 70}")
    print("GERANDO GRÁFICOS DE ZIPF")
    print(f"{ '=' * 70}")
    
    print("\n[1/3] Gráfico Combinado (Palavras + Pares):")
    plotar_zipf(dados_para_grafico_combinado, dados_global_combinado, 
                arquivo_saida="analise_resultados/zipf/zipf_combinado.png", 
                titulo="Palavras + Pares (Sistema Real)")
    
    print("\n[2/3] Gráfico de Palavras:")
    plotar_zipf(dados_para_grafico_uni, dados_global_uni, 
                arquivo_saida="analise_resultados/zipf/zipf_palavras.png", 
                titulo="Palavras")
    
    print("\n[3/3] Gráfico de Pares:")
    plotar_zipf(dados_para_grafico_bi, dados_global_bi, 
                arquivo_saida="analise_resultados/zipf/zipf_pares.png", 
                titulo="Pares de Palavras")
    
    print(f"\n{'=' * 70}")
    print("ESTATÍSTICAS FINAIS")
    print(f"{ '=' * 70}")
    print(f"\nTOTAIS GERAIS:")
    print(f"   • Palavras distintas:       {len(palavras_global):>12,}")
    print(f"   • Ocorrências de palavras:  {total_ocorrencias_uni:>12,}")
    print(f"   • Pares distintos:          {len(pares_global):>12,}")
    print(f"   • Ocorrências de pares:     {total_ocorrencias_bi:>12,}")
    print(f"   • N-gramas combinados:      {len(ngramas_global_combinados):>12,}")
    print(f"   • Ocorrências combinadas:   {total_ocorrencias_combinadas:>12,}")
    
    print(f"\nTOP 20 PALAVRAS:")
    for i, (token, freq) in enumerate(palavras_global.most_common(20), 1):
        porcentagem = (freq / total_ocorrencias_uni) * 100
        print(f"   {i:2d}. {token:25s} {freq:>12,} ({porcentagem:5.2f}%)")
    
    print(f"\nTOP 20 PARES DE PALAVRAS:")
    for i, (par, freq) in enumerate(pares_global.most_common(20), 1):
        porcentagem = (freq / total_ocorrencias_bi) * 100
        print(f"   {i:2d}. {par:45s} {freq:>12,} ({porcentagem:5.2f}%)")
    
    print(f"\nTOP 20 N-GRAMAS COMBINADOS (Sistema Real):")
    for i, (ngrama, freq) in enumerate(ngramas_global_combinados.most_common(20), 1):
        porcentagem = (freq / total_ocorrencias_combinadas) * 100
        tipo = "pal" if " " not in ngrama else "par"
        print(f"   {i:2d}. [{tipo}] {ngrama:43s} {freq:>12,} ({porcentagem:5.2f}%)")
    
    print(f"\n{'=' * 70}\n")
    
    return True


if __name__ == "__main__":
    print("\nAnálise de Distribuição de Palavras e Pares de Palavras")
    print("   Usa dados salvos de resultados_benchmark/Counter\n")
    
    sucesso = analisar_zipf()
    
    if sucesso:
        print("Análise concluída com sucesso!\n")
    else:
        print("Erro na análise.\n")
