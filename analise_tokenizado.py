"""Análise estatística do corpus tokenizado."""
from pathlib import Path
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv
import numpy as np
import gc

load_dotenv()


def analisar_corpus():
    """Analisa estatísticas do corpus tokenizado."""
    
    # Configuração de processamento em lotes
    LOTE_DOCUMENTOS = 5000  # Processar documentos em lotes de 5000
    
    print("=" * 70)
    print("ANÁLISE ESTATÍSTICA DO CORPUS TOKENIZADO")
    print("=" * 70)
    
    janela = int(os.getenv('JANELA', 3))
    print(f"\nConfiguração: JANELA = {janela}")
    
    tokenizado_dir = Path("corpus/tokenizado")
    
    if not tokenizado_dir.exists():
        print("\nErro: Diretório corpus/tokenizado não encontrado!")
        print("   Execute primeiro o script de tokenização.")
        return False
    
    partes = sorted(list(tokenizado_dir.glob("parte_*.txt")))
    if not partes:
        print("\nErro: Nenhuma parte tokenizada encontrada!")
        return False
    
    print(f"\n✓ Encontradas {len(partes)} partes tokenizadas")
    print("   Cada documento é processado INDIVIDUALMENTE.")
    print(f"   Pares de palavras extraídos apenas dentro de cada documento (janela={janela}).")
    print("   Pares de palavras NÃO cruzam fronteiras entre documentos.\n")
    print("\nIniciando análise...\n")
    
    stats_globais = {
        'total_partes': len(partes),
        'total_documentos': 0,
        'total_tokens': 0,
        'total_palavras': 0,
        'total_pares': 0,
        'total_palavras_unicas': 0,
        'total_pares_unicos': 0,
        'tokens_por_parte': [],
        'palavras_por_parte': [],
        'pares_por_parte': [],
        'tokens_por_documento': [],
        'chars_por_documento': []
    }
    
    stats_por_parte = []
    
    # Estatísticas de todos os documentos individuais
    todos_documentos = []
    
    # Conjuntos globais para tracking (usaremos hashes para economizar memória)
    # Vamos usar um set global mas limpar após cada parte
    palavras_globais_vistas = set()
    pares_globais_vistos = set()
    total_pares_global = 0  # Conta TODAS as ocorrências de pares no corpus
    
    # Processar cada parte
    for idx, arquivo in enumerate(partes, 1):
        print(f"[{idx:02d}/{len(partes)}] Analisando {arquivo.name}...")
        
        # Ler documentos individuais (cada linha é um documento)
        with open(arquivo, 'r', encoding='utf-8') as f:
            documentos = [linha.strip() for linha in f if linha.strip()]
        
        num_docs = len(documentos)
        if num_docs == 0:
            print(f"  Arquivo vazio, pulando...")
            continue
        
        print(f"  {num_docs:,} documentos encontrados")
        
        # Estatísticas para esta parte
        tokens_parte_total = 0
        total_pares_parte = 0  # Conta TODAS as ocorrências de pares nesta parte
        docs_stats_parte = []
        
        # Processar cada documento individualmente para estatísticas
        for doc in tqdm(documentos, desc=f"  Docs individuais", 
                       unit="doc", ncols=70, leave=False):
            tokens_doc = doc.split(';')
            num_tokens_doc = len(tokens_doc)
            num_chars_doc = len(doc)
            
            # Guardar estatísticas do documento
            doc_stat = {
                'num_tokens': num_tokens_doc,
                'num_chars': num_chars_doc,
                'parte': arquivo.name
            }
            docs_stats_parte.append(doc_stat)
            todos_documentos.append(doc_stat)
            
            # Atualizar contadores globais
            tokens_parte_total += num_tokens_doc
            stats_globais['tokens_por_documento'].append(num_tokens_doc)
            stats_globais['chars_por_documento'].append(num_chars_doc)
        
        # OTIMIZAÇÃO: Processar em lotes para economizar memória
        print(f"  Processando em lotes de {LOTE_DOCUMENTOS:,} documentos...")
        
        palavras_parte = set()
        pares_parte = set()
        
        # Processar documentos em lotes
        for lote_inicio in range(0, num_docs, LOTE_DOCUMENTOS):
            lote_fim = min(lote_inicio + LOTE_DOCUMENTOS, num_docs)
            lote_docs = documentos[lote_inicio:lote_fim]
            
            print(f"    Lote [{lote_inicio+1:,}-{lote_fim:,}]...", end=' ')
            
            # Processar cada documento do lote INDIVIDUALMENTE
            for doc in lote_docs:
                tokens_doc = doc.split(";")
                num_tokens_doc = len(tokens_doc)
                
                # Processar tokens: palavras e pares de palavras
                for i in range(num_tokens_doc):
                    # Palavra
                    palavras_parte.add(tokens_doc[i])
                    
                    # Pares de palavras apenas dentro deste documento (não cruza fronteiras)
                    if i < num_tokens_doc - 1:
                        fim_janela = min(i + janela, num_tokens_doc)
                        for j in range(i + 1, fim_janela):
                            # Ordenar tokens do par
                            token1, token2 = tokens_doc[i], tokens_doc[j]
                            if token1 <= token2:
                                par = f"{token1} {token2}"
                            else:
                                par = f"{token2} {token1}"
                            pares_parte.add(par)
                            total_pares_parte += 1  # Contar cada ocorrência
            
            print(f"✓ ({len(palavras_parte):,} pal, {len(pares_parte):,} pares dist)")
            
            # Liberar memória do lote
            del lote_docs
            gc.collect()
        
        # Atualizar contadores globais
        for palavra in palavras_parte:
            palavras_globais_vistas.add(palavra)
        
        for par in pares_parte:
            pares_globais_vistos.add(par)
        
        # Calcular estatísticas dos documentos desta parte
        tokens_docs_parte = [d['num_tokens'] for d in docs_stats_parte]
        chars_docs_parte = [d['num_chars'] for d in docs_stats_parte]
        
        media_tokens_doc_parte = np.mean(tokens_docs_parte)
        std_tokens_doc_parte = np.std(tokens_docs_parte, ddof=1) if num_docs > 1 else 0.0
        media_chars_doc_parte = np.mean(chars_docs_parte)
        std_chars_doc_parte = np.std(chars_docs_parte, ddof=1) if num_docs > 1 else 0.0
        
        # Salvar estatísticas da parte
        total_distintos_parte = len(palavras_parte) + len(pares_parte)
        total_tokens_parte = tokens_parte_total + total_pares_parte  # palavras + TODAS as ocorrências de pares
        total_pares_global += total_pares_parte  # Adicionar ao contador global
        stats_parte = {
            'arquivo': arquivo.name,
            'num_documentos': num_docs,
            'total_tokens': total_tokens_parte,
            'total_palavras': tokens_parte_total,
            'total_pares': total_pares_parte,
            'palavras_distintas': len(palavras_parte),
            'pares_distintos': len(pares_parte),
            'total_distintos': total_distintos_parte,
            'media_tokens_por_doc': float(media_tokens_doc_parte),
            'std_tokens_por_doc': float(std_tokens_doc_parte),
            'media_chars_por_doc': float(media_chars_doc_parte),
            'std_chars_por_doc': float(std_chars_doc_parte)
        }
        stats_por_parte.append(stats_parte)
        
        # Atualizar estatísticas globais
        stats_globais['total_documentos'] += num_docs
        stats_globais['total_tokens'] += total_tokens_parte
        stats_globais['total_palavras'] += tokens_parte_total
        stats_globais['total_pares'] += total_pares_parte
        stats_globais['tokens_por_parte'].append(total_tokens_parte)
        stats_globais['palavras_por_parte'].append(len(palavras_parte))
        stats_globais['pares_por_parte'].append(len(pares_parte))
        
        # Mostrar resultados da parte
        print(f"  ✓ Documentos: {num_docs:,}")
        print(f"  ✓ Tokens totais: {total_tokens_parte:,} (palavras: {tokens_parte_total:,}, pares: {total_pares_parte:,})")
        print(f"  ✓ Média tokens/doc: {media_tokens_doc_parte:.1f} ± {std_tokens_doc_parte:.1f}")
        print(f"  ✓ Média chars/doc: {media_chars_doc_parte:.1f} ± {std_chars_doc_parte:.1f}")
        print(f"  ✓ Palavras distintas: {len(palavras_parte):,}")
        print(f"  ✓ Pares distintos: {len(pares_parte):,}")
        print(f"  ✓ Total distintos (pal+pares): {total_distintos_parte:,}")
        print()
        
        # Liberar memória desta parte
        del palavras_parte
        del pares_parte
        del documentos
        del docs_stats_parte
        del tokens_docs_parte
        del chars_docs_parte
        gc.collect()
    
    # Calcular médias e desvios padrão GLOBAIS (de todos os documentos)
    media_tokens_doc_global = np.mean(stats_globais['tokens_por_documento'])
    std_tokens_doc_global = np.std(stats_globais['tokens_por_documento'], ddof=1)
    
    media_chars_doc_global = np.mean(stats_globais['chars_por_documento'])
    std_chars_doc_global = np.std(stats_globais['chars_por_documento'], ddof=1)
    
    # Calcular médias e desvios padrão POR PARTE
    media_tokens_parte = np.mean(stats_globais['tokens_por_parte'])
    std_tokens_parte = np.std(stats_globais['tokens_por_parte'], ddof=1)
    
    media_palavras = np.mean(stats_globais['palavras_por_parte'])
    std_palavras = np.std(stats_globais['palavras_por_parte'], ddof=1)
    
    media_pares = np.mean(stats_globais['pares_por_parte'])
    std_pares = np.std(stats_globais['pares_por_parte'], ddof=1)
    
    # Calcular desvio de cada parte em relação à média global
    for stats in stats_por_parte:
        stats['desvio_da_media_tokens'] = abs(stats['total_tokens'] - media_tokens_parte)
        stats['desvio_da_media_palavras'] = abs(stats['palavras_distintas'] - media_palavras)
        stats['desvio_da_media_pares'] = abs(stats['pares_distintos'] - media_pares)
    
    # Calcular total de distintos usando os sets globais
    stats_globais['total_palavras_unicas'] = len(palavras_globais_vistas)
    stats_globais['total_pares_unicos'] = len(pares_globais_vistos)
    total_distintos_global = stats_globais['total_palavras_unicas'] + stats_globais['total_pares_unicos']
    media_distintos_por_parte = sum(s['total_distintos'] for s in stats_por_parte) / len(stats_por_parte)
    
    # Limpar sets globais - não precisamos mais
    del palavras_globais_vistas
    del pares_globais_vistos
    gc.collect()
    
    # Exibir relatório final
    print("=" * 70)
    print("RELATÓRIO FINAL")
    print("=" * 70)
    
    print("\nTOTAIS GERAIS:")
    print(f"   • Total de partes:                    {stats_globais['total_partes']:>12,}")
    print(f"   • Total de documentos:                {stats_globais['total_documentos']:>12,}")
    print(f"   • Total de tokens (pal + pares):      {stats_globais['total_tokens']:>12,}")
    print(f"   • Total de palavras:                  {stats_globais['total_palavras']:>12,}")
    print(f"   • Total de pares:                     {stats_globais['total_pares']:>12,}")
    print(f"   • Palavras distintas:                 {stats_globais['total_palavras_unicas']:>12,}")
    print(f"   • Pares de palavras distintos:        {stats_globais['total_pares_unicos']:>12,}")
    print(f"   • Total distintos (pal + pares):      {total_distintos_global:>12,}")
    
    print("\nESTATÍSTICAS DE DOCUMENTOS (todos):")
    print(f"   • Tokens por documento:       {media_tokens_doc_global:>12.1f} ± {std_tokens_doc_global:>10.1f}")
    print(f"   • Caracteres por documento:   {media_chars_doc_global:>12.1f} ± {std_chars_doc_global:>10.1f}")
    
    print("\nESTATÍSTICAS POR PARTE:")
    print(f"   • Tokens por parte (pal+pares): {media_tokens_parte:>10.1f} ± {std_tokens_parte:>10.1f}")
    print(f"   • Palavras distintas:           {media_palavras:>10.1f} ± {std_palavras:>10.1f}")
    print(f"   • Pares distintos:              {media_pares:>10.1f} ± {std_pares:>10.1f}")
    print(f"   • Total distintos por parte:  {media_distintos_por_parte:>12.1f}")
    
    print("\nDETALHAMENTO POR PARTE:")
    print(f"{'Parte':<12} {'Docs':>7} {'Tokens':>12} {'Palavras':>11} {'Pares':>11} {'Tok/Doc':>10} {'Chars/Doc':>11} {'Pal.Dist':>10} {'Par.Dist':>10} {'Total':>11}")
    print("-" * 140)
    for stats in stats_por_parte:
        print(f"{stats['arquivo']:<12} "
              f"{stats['num_documentos']:>7,} "
              f"{stats['total_tokens']:>12,} "
              f"{stats['total_palavras']:>11,} "
              f"{stats['total_pares']:>11,} "
              f"{stats['media_tokens_por_doc']:>7.1f}±{stats['std_tokens_por_doc']:<3.0f} "
              f"{stats['media_chars_por_doc']:>8.1f}±{stats['std_chars_por_doc']:<3.0f} "
              f"{stats['palavras_distintas']:>10,} "
              f"{stats['pares_distintos']:>10,} "
              f"{stats['total_distintos']:>11,}")
    
    # Salvar relatório em JSON
    relatorio_path = Path("corpus/relatorio_analise.json")
    
    relatorio = {
        'configuracao': {
            'janela_bigramas': janela
        },
        'resumo_geral': {
            'total_partes': stats_globais['total_partes'],
            'total_documentos': stats_globais['total_documentos'],
            'total_tokens': stats_globais['total_tokens'],
            'total_palavras': stats_globais['total_palavras'],
            'total_pares': stats_globais['total_pares'],
            'palavras_distintas': stats_globais['total_palavras_unicas'],
            'pares_palavras_distintos': stats_globais['total_pares_unicos'],
            'total_distintos': total_distintos_global
        },
        'estatisticas_documentos': {
            'media_tokens_por_doc': float(media_tokens_doc_global),
            'std_tokens_por_doc': float(std_tokens_doc_global),
            'media_chars_por_doc': float(media_chars_doc_global),
            'std_chars_por_doc': float(std_chars_doc_global)
        },
        'estatisticas_partes': {
            'media_tokens_por_parte': float(media_tokens_parte),
            'std_tokens_por_parte': float(std_tokens_parte),
            'media_palavras_por_parte': float(media_palavras),
            'std_palavras_por_parte': float(std_palavras),
            'media_pares_por_parte': float(media_pares),
            'std_pares_por_parte': float(std_pares),
            'media_distintos_por_parte': media_distintos_por_parte
        },
        'por_parte': stats_por_parte,
        'todos_documentos': todos_documentos
    }
    
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    
    print(f"\nRelatório salvo em: {relatorio_path.absolute()}")
    print("=" * 70)
    print()
    
    return True


if __name__ == "__main__":
    print("\nAnalisador de Corpus Tokenizado")
    print("   Estatísticas de palavras e pares de palavras\n")
    
    sucesso = analisar_corpus()
    
    if sucesso:
        print("Análise concluída com sucesso!\n")
    else:
        print("Erro na análise.\n")
