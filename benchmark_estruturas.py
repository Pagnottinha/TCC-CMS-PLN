#!/usr/bin/env python3
"""Benchmark de estruturas de contagem probabilísticas."""

import os
import time
import json
import csv
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from factory_estruturas import (
    criar_estrutura, 
    incrementar_estrutura, 
    estimar_estrutura,
    obter_classe_estrutura,
    CONFIGS_PADRAO as CONFIGS
)

load_dotenv()
SEED = int(os.getenv('SEED', 42))
JANELA = int(os.getenv('JANELA', 3))

CORPUS_DIR = Path('corpus/tokenizado')
OUTPUT_DIR = Path('resultados_benchmark')
OUTPUT_DIR.mkdir(exist_ok=True)
RELATORIO_ANALISE = Path('corpus/relatorio_analise.json')


def limpar_resultados():
    """Remove diretório de resultados existente."""
    if OUTPUT_DIR.exists():
        print(f"\nRemovendo resultados anteriores: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        print("  ✓ Diretório removido")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("  ✓ Diretório criado")


def carregar_info_corpus():
    """Carrega informações do relatório de análise do corpus."""
    if RELATORIO_ANALISE.exists():
        with open(RELATORIO_ANALISE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            resumo = data['resumo_geral']
            
            # Extrair total_tokens de cada parte (já inclui palavras + pares)
            tamanhos_por_parte = []
            for parte in data.get('por_parte', []):
                tamanhos_por_parte.append(parte['total_tokens'])
            
            return {
                'resumo': resumo,
                'tamanhos_por_parte': tamanhos_por_parte
            }
    return None


def ler_documentos(arquivo: Path) -> List[str]:
    """Lê documentos de um arquivo tokenizado."""
    with open(arquivo, 'r', encoding='utf-8') as f:
        documentos = [linha.strip() for linha in f if linha.strip()]
    return documentos



def processar_parte(arquivo: Path, estruturas: Dict, 
                    counter_palavras: Counter, counter_pares: Counter,
                    tamanho_estimado: int) -> Dict:
    """Processa uma parte do corpus e retorna tempos de inserção."""
    print(f"\n{'='*80}")
    print(f"Processando: {arquivo.name}")
    print('=' * 80)
    
    print("Lendo documentos...", end=' ', flush=True)
    documentos = ler_documentos(arquivo)
    num_docs = len(documentos)
    print(f"✓ {num_docs:,} documentos")
    
    if num_docs == 0:
        print("  Arquivo vazio, pulando...")
        return {}
    
    tempos_arrays = {
        'Counter': np.zeros(tamanho_estimado, dtype=np.float64),
    }
    
    for nome in estruturas.keys():
        tempos_arrays[nome] = np.zeros(tamanho_estimado, dtype=np.float64)
    
    idx = 0
    
    for doc in tqdm(documentos, desc=f"  {arquivo.name}", unit="doc", ncols=70, 
                    leave=False):
        tokens_doc = doc.split(';')
        num_tokens_doc = len(tokens_doc)
        
        for i in range(num_tokens_doc):
            token = tokens_doc[i]
            
            inicio = time.perf_counter()
            counter_palavras[token] += 1
            fim = time.perf_counter()
            tempos_arrays['Counter'][idx] = fim - inicio
            
            for nome, estrutura in estruturas.items():
                inicio = time.perf_counter()
                incrementar_estrutura(estrutura, token)
                fim = time.perf_counter()
                tempos_arrays[nome][idx] = fim - inicio
            
            idx += 1
            
            if i < num_tokens_doc - 1:
                fim_janela = min(i + JANELA, num_tokens_doc)
                for j in range(i + 1, fim_janela):
                    token1, token2 = tokens_doc[i], tokens_doc[j]
                    if token1 <= token2:
                        par = f"{token1} {token2}"
                    else:
                        par = f"{token2} {token1}"
                    
                    inicio = time.perf_counter()
                    counter_pares[par] += 1
                    fim = time.perf_counter()
                    tempos_arrays['Counter'][idx] = fim - inicio
                    
                    for nome, estrutura in estruturas.items():
                        inicio = time.perf_counter()
                        incrementar_estrutura(estrutura, par)
                        fim = time.perf_counter()
                        tempos_arrays[nome][idx] = fim - inicio
                    
                    idx += 1
    
    print(f"  ✓ Counter: {len(counter_palavras):,} palavras únicas")
    print(f"  ✓ Counter: {len(counter_pares):,} pares únicos")
    
    tempos_truncados = {}
    for nome, arr in tempos_arrays.items():
        tempos_truncados[nome] = arr[:idx]
    
    return tempos_truncados


def calcular_estatisticas(tempos_array: np.ndarray) -> Dict:
    """Calcula média, desvio padrão e mediana dos tempos."""
    # Retorna 0 se o array estiver vazio para evitar warnings do numpy
    if len(tempos_array) == 0:
        return {
            'media': 0.0, 'mediana': 0.0, 'desvio_padrao': 0.0, 
            'total': 0.0, 'min': 0.0, 'max': 0.0, 'quantidade': 0
        }
    return {
        'media': float(np.mean(tempos_array)),
        'mediana': float(np.median(tempos_array)),
        'desvio_padrao': float(np.std(tempos_array)),
        'total': float(np.sum(tempos_array)),
        'min': float(np.min(tempos_array)),
        'max': float(np.max(tempos_array)),
        'quantidade': len(tempos_array)
    }


def calcular_mre(estrutura, counter_palavras: Counter, counter_pares: Counter) ->Tuple[Dict, np.ndarray]:
    """Calcula Mean Relative Error (MRE) e tempos de consulta."""
    num_consultas = len(counter_palavras) + len(counter_pares)
    tempos_consulta = np.zeros(num_consultas, dtype=np.float64)
    idx = 0
    
    erros_palavras = []
    erros_pares = []
    
    for token, contagem_real in counter_palavras.items():
        inicio = time.perf_counter()
        contagem_estimada = estimar_estrutura(estrutura, token)
        fim = time.perf_counter()
        
        if idx < len(tempos_consulta):
            tempos_consulta[idx] = fim - inicio
        idx += 1
        
        if contagem_real > 0:  # Evitar divisão por zero
            erro_relativo = abs(contagem_estimada - contagem_real) / contagem_real
            erros_palavras.append(erro_relativo)
    
    for par, contagem_real in counter_pares.items():
        inicio = time.perf_counter()
        contagem_estimada = estimar_estrutura(estrutura, par)
        fim = time.perf_counter()

        if idx < len(tempos_consulta):
            tempos_consulta[idx] = fim - inicio
        idx += 1

        if contagem_real > 0:
            erro_relativo = abs(contagem_estimada - contagem_real) / contagem_real
            erros_pares.append(erro_relativo)
    
    mre_palavras = float(np.mean(erros_palavras)) if erros_palavras else 0.0
    mre_pares = float(np.mean(erros_pares)) if erros_pares else 0.0
    mre_total = float(np.mean(erros_palavras + erros_pares)) if (erros_palavras or erros_pares) else 0.0
    
    mre_metrics = {
        'mre_palavras': mre_palavras,
        'mre_pares': mre_pares,
        'mre_total': mre_total,
        'num_palavras': len(erros_palavras),
        'num_pares': len(erros_pares)
    }
    
    return (mre_metrics, tempos_consulta[:idx])


def salvar_estrutura(estrutura, nome: str, parte: int):
    """Salva a estrutura em arquivo."""
    pasta_estrutura = OUTPUT_DIR / nome
    pasta_estrutura.mkdir(parents=True, exist_ok=True)
    
    nome_arquivo = pasta_estrutura / f"parte{parte:02d}"
    estrutura.salvar(str(nome_arquivo))


def imprimir_resumo_mre(mre_totais):
    """Imprime o resumo do MRE."""
    print("\n" + "="*80)
    print("RESUMO DO MRE (Mean Relative Error)")
    print("="*80)
    print(f"\n{'Estrutura':<20} {'MRE Total':<22} {'MRE Palavras':<22} {'MRE Pares':<22}")
    print("-" * 86)
    
    for nome in sorted(mre_totais.keys()):
        mre = mre_totais[nome]
        print(f"{nome:<20} {mre['mre_total']:<22.6f} {mre['mre_palavras']:<22.6f} {mre['mre_pares']:<22.6f}")

def imprimir_resumo_tempo(titulo: str, estatisticas_totais: dict):
    """Imprime um resumo dos tempos de execução."""
    print("\n" + "="*80)
    print(titulo)
    print("="*80)
    print(f"\n{'Estrutura':<20} {'Média (s)':<22} {'Mediana (s)':<22} {'Desvio Padrão (s)':<22} {'Min (s)':<22} {'Max (s)':<22} {'Total (s)':<22}")
    print("-" * 171)
    
    for nome in sorted(estatisticas_totais.keys()):
        stats = estatisticas_totais[nome]
        tipo = "Pal+Pares" if "insercao" in titulo.lower() else ""
        print(f"{nome:<20} {stats['media']:<22.6e} {stats['mediana']:<22.6e} {stats['desvio_padrao']:<22.6e} {stats['min']:<22.6e} {stats['max']:<22.6e} {stats['total']:<22.2f}")

def main():

    print("="*80)
    print("BENCHMARK DE ESTRUTURAS DE CONTAGEM")
    print("="*80)
    print(f"SEED: {SEED}")
    print(f"JANELA: {JANELA}")
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Saída: {OUTPUT_DIR}")
    
    relatorio_path = OUTPUT_DIR / 'relatorio_completo.json'
    pular_insercao = False
    relatorio = {}
    
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print("\n" + "="*80)
        print("RESULTADOS ANTERIORES DETECTADOS")
        print("="*80)
        print(f"\nO diretório {OUTPUT_DIR} contém arquivos de execuções anteriores.")
        print("\nOpções:")
        print("  [1] Limpar tudo e executar benchmark do zero")
        print("  [2] Pular inserção e calcular apenas MRE (usa arquivos existentes)")
        print("  [3] Apenas exibir o relatório de resultados existente")
        print("  [4] Cancelar") # Esta será a nova opção [3]
        
        escolha = input("\nEscolha uma opção (1/2/3/4): ").strip()
        
        if escolha == '1':
            limpar_resultados()
            print("\n✓ Executando benchmark completo do zero...")
        elif escolha == '2': # Esta será a nova opção [2] para pular inserção
            if relatorio_path.exists():
                pular_insercao = True
                print("\n✓ Pulando fase de inserção e carregando relatório existente...")
                with open(relatorio_path, 'r', encoding='utf-8') as f:
                    relatorio = json.load(f)
                arquivos = sorted(CORPUS_DIR.glob('parte_*.txt'))
            else:
                print(f"\n✗ Erro: {relatorio_path} não encontrado!")
                print("   Não é possível pular a inserção sem o relatório.")
                return
        elif escolha == '3': # Esta será a nova opção [3] para exibir relatório existente
            if relatorio_path.exists():
                print("\n✓ Carregando e exibindo relatório existente...")
                with open(relatorio_path, 'r', encoding='utf-8') as f:
                    relatorio_existente = json.load(f)
                
                if 'estatisticas_insercao_totais' in relatorio_existente:
                    imprimir_resumo_tempo("RESUMO DOS TEMPOS DE INSERÇÃO (segundos)", relatorio_existente['estatisticas_insercao_totais'])
                
                if 'mre_totais' in relatorio_existente:
                    imprimir_resumo_mre(relatorio_existente['mre_totais'])
                
                if 'estatisticas_consulta_totais' in relatorio_existente:
                    imprimir_resumo_tempo("RESUMO DOS TEMPOS DE CONSULTA (segundos)", relatorio_existente['estatisticas_consulta_totais'])
            else:
                print(f"\n✗ Erro: {relatorio_path} não encontrado!")
            return
        else:
            print("\n✓ Operação cancelada.")
            return
    else:
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    if not pular_insercao:
        print("\nCarregando informações do corpus...")
        info_corpus = carregar_info_corpus()
        if info_corpus:
            tamanhos_por_parte = info_corpus['tamanhos_por_parte']
            print(f"  ✓ Tamanhos por parte carregados: {len(tamanhos_por_parte)} partes")
        else:
            print("  ! Relatório não encontrado, usando estimativa padrão")
            tamanhos_por_parte = [100_000_000] * 10
        
        arquivos = sorted(CORPUS_DIR.glob('parte_*.txt'))
        tempos_por_parte = []
        
        stats_acumuladas_por_metrica = {}
        for nome in ['Counter'] + list(CONFIGS.keys()):
            stats_acumuladas_por_metrica[nome] = {
                'medias': [], 'medianas': [], 'desvios_padrao': [], 'totais': [],
                'mins': [], 'maxs': [], 'quantidades': []
            }
        
        for i, arquivo in enumerate(arquivos, 1):
            print(f"\nCriando estruturas para parte {i}...")
            estruturas = {}
            for nome, config in CONFIGS.items():
                print(f"  {nome}...", end=' ', flush=True)
                estruturas[nome] = criar_estrutura(nome, config)
                print("✓")
            
            counter_palavras = Counter()
            counter_pares = Counter()
            
            tamanho_parte = tamanhos_por_parte[i-1] if i-1 < len(tamanhos_por_parte) else 100_000_000
            print(f"  Tamanho esperado: {tamanho_parte:,} tokens")
            
            tempos = processar_parte(arquivo, estruturas, counter_palavras, counter_pares, tamanho_parte)
            part_stats_all_structures = {chave: calcular_estatisticas(valores) for chave, valores in tempos.items()}
            tempos_por_parte.append({
                'parte': i,
                'arquivo': arquivo.name,
                'estatisticas': part_stats_all_structures
            })
            
            for chave, stats_da_parte in part_stats_all_structures.items():
                stats = stats_acumuladas_por_metrica[chave]
                stats['medias'].append(stats_da_parte['media'])
                stats['medianas'].append(stats_da_parte['mediana'])
                stats['desvios_padrao'].append(stats_da_parte['desvio_padrao'])
                stats['totais'].append(stats_da_parte['total'])
                stats['mins'].append(stats_da_parte['min'])
                stats['maxs'].append(stats_da_parte['max'])
                stats['quantidades'].append(stats_da_parte['quantidade'])
            
            print(f"\nSalvando estruturas da parte {i}:")
            for nome, estrutura in estruturas.items():
                salvar_estrutura(estrutura, nome, i)
            
            print(f"\nSalvando contadores da parte {i}:")
            pasta_counter = OUTPUT_DIR / 'Counter'
            pasta_counter.mkdir(parents=True, exist_ok=True)
            path_palavras = pasta_counter / f'palavras_parte{i:02d}.csv'
            with open(path_palavras, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['token', 'freq'])
                writer.writerows(counter_palavras.items())
            print(f"    {path_palavras.name}: {len(counter_palavras):,} únicos")

            path_pares = pasta_counter / f'pares_parte{i:02d}.csv'
            with open(path_pares, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['token', 'freq'])
                writer.writerows(counter_pares.items())
            print(f"    {path_pares.name}: {len(counter_pares):,} únicos")
        
        estatisticas_totais_insercao = {}
        for chave, acumulados in stats_acumuladas_por_metrica.items():
            if acumulados['quantidades']:
                total_quantidade = sum(acumulados['quantidades'])
                estatisticas_totais_insercao[chave] = {
                    'media': float(np.mean(acumulados['medias'])),
                    'mediana': float(np.mean(acumulados['medianas'])),
                    'desvio_padrao': float(np.mean(acumulados['desvios_padrao'])),
                    'total': float(np.sum(acumulados['totais'])),
                    'min': float(np.min(acumulados['mins'])),
                    'max': float(np.max(acumulados['maxs'])),
                    'quantidade': total_quantidade
                }
        relatorio['estatisticas_insercao_totais'] = estatisticas_totais_insercao
        
    if 'mre_totais' not in relatorio:
        arquivos = sorted(CORPUS_DIR.glob('parte_*.txt'))
        print("\n" + "="*80)
        print("CALCULANDO MRE (Mean Relative Error) E TEMPOS DE CONSULTA")
        print("="*80)
        
        mre_por_parte = []
        stats_consulta_acumuladas = {}
        for nome in CONFIGS.keys():
            stats_consulta_acumuladas[nome] = {
                'medias': [], 'medianas': [], 'desvios_padrao': [], 'totais': [],
                'mins': [], 'maxs': [], 'quantidades': []
            }
        
        for i, arquivo in enumerate(arquivos, 1):
            print(f"\n[Parte {i:02d}] Carregando arquivos...")
            pasta_counter = OUTPUT_DIR / 'Counter'
            counter_palavras = Counter()
            with open(pasta_counter / f'palavras_parte{i:02d}.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    counter_palavras[row[0]] = int(row[1])

            counter_pares = Counter()
            with open(pasta_counter / f'pares_parte{i:02d}.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    counter_pares[row[0]] = int(row[1])
            print(f"  ✓ Counters carregados: {len(counter_palavras):,} palavras, {len(counter_pares):,} pares")
            
            mre_parte = {'parte': i, 'estruturas': {}}
            for nome, config in CONFIGS.items():
                print(f"  Calculando MRE e tempo de consulta: {nome}...", end=' ', flush=True)
                pasta_estrutura = OUTPUT_DIR / nome
                classe_estrutura = obter_classe_estrutura(nome)
                estrutura = classe_estrutura.carregar(str(pasta_estrutura / f"parte{i:02d}"))
                
                mre_metrics, tempos_consulta = calcular_mre(estrutura, counter_palavras, counter_pares)
                stats_consulta = calcular_estatisticas(tempos_consulta)
                
                mre_parte['estruturas'][nome] = {'mre': mre_metrics, 'estatisticas_consulta': stats_consulta}
                
                stats = stats_consulta_acumuladas[nome]
                stats['medias'].append(stats_consulta['media'])
                stats['medianas'].append(stats_consulta['mediana'])
                stats['desvios_padrao'].append(stats_consulta['desvio_padrao'])
                stats['totais'].append(stats_consulta['total'])
                stats['mins'].append(stats_consulta['min'])
                stats['maxs'].append(stats_consulta['max'])
                stats['quantidades'].append(stats_consulta['quantidade'])
                print(f"MRE={mre_metrics['mre_total']:.6f}")
            mre_por_parte.append(mre_parte)
        
        mre_totais = {}
        for nome in CONFIGS.keys():
            mre_totais[nome] = {
                'mre_palavras': float(np.mean([p['estruturas'][nome]['mre']['mre_palavras'] for p in mre_por_parte])),
                'mre_pares': float(np.mean([p['estruturas'][nome]['mre']['mre_pares'] for p in mre_por_parte])),
                'mre_total': float(np.mean([p['estruturas'][nome]['mre']['mre_total'] for p in mre_por_parte]))
            }
        
        estatisticas_consulta_totais = {}
        for nome, acumulados in stats_consulta_acumuladas.items():
            if acumulados['quantidades']:
                total_quantidade = sum(acumulados['quantidades'])
                estatisticas_consulta_totais[nome] = {
                    'media': float(np.mean(acumulados['medias'])),
                    'mediana': float(np.mean(acumulados['medianas'])),
                    'desvio_padrao': float(np.mean(acumulados['desvios_padrao'])),
                    'total': float(np.sum(acumulados['totais'])),
                    'min': float(np.min(acumulados['mins'])),
                    'max': float(np.max(acumulados['maxs'])),
                    'quantidade': total_quantidade
                }
        
        relatorio.update({
            'mre_por_parte': mre_por_parte,
            'mre_totais': mre_totais,
            'estatisticas_consulta_totais': estatisticas_consulta_totais
        })

    print("\n" + "="*80)
    print("SALVANDO E EXIBINDO RESULTADOS FINAIS")
    print("="*80)

    relatorio['configuracao'] = {
        'seed': SEED, 'janela': JANELA, 'estruturas': CONFIGS
    }
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        json.dump(relatorio, f, indent=2)
    print(f"  ✓ Relatório unificado salvo: {relatorio_path}")

    if 'estatisticas_insercao_totais' in relatorio:
        imprimir_resumo_tempo("RESUMO DOS TEMPOS DE INSERÇÃO (segundos)", relatorio['estatisticas_insercao_totais'])
    
    if 'mre_totais' in relatorio:
        imprimir_resumo_mre(relatorio['mre_totais'])
    
    if 'estatisticas_consulta_totais' in relatorio:
        imprimir_resumo_tempo("RESUMO DOS TEMPOS DE CONSULTA (segundos)", relatorio['estatisticas_consulta_totais'])

    print("\n" + "="*80)
    print("CONCLUÍDO!")
    print("="*80)
    print(f"Resultados salvos em: {OUTPUT_DIR.absolute()}")

if __name__ == '__main__':
    main()