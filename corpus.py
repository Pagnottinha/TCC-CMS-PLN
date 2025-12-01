
"""
Script para preparar corpus dividido em 10 partes
Armazena documentos completos
"""
from pathlib import Path
import random
from datasets import load_dataset
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
SEED = int(os.getenv('SEED', 42))

def preparar_corpus():
    """Baixa o corpus Carolina e divide em 10 partes aleatórias."""
    
    print("=" * 60)
    print("PREPARAÇÃO DO CORPUS (documentos completos)")
    print("=" * 60)
    
    corpus_dir = Path("corpus")
    corpus_dir.mkdir(exist_ok=True)
    
    partes_existentes = list(corpus_dir.glob("parte_*.txt"))
    if partes_existentes:
        print(f"\n✓ Encontradas {len(partes_existentes)} partes já criadas:")
        for parte in sorted(partes_existentes):
            linhas = sum(1 for _ in open(parte, encoding='utf-8'))
            tamanho_mb = parte.stat().st_size / (1024 * 1024)
            print(f"  - {parte.name}: {linhas:6d} documentos ({tamanho_mb:.2f} MB)")
        
        print(f"\nO que deseja fazer?")
        print(f"  1) Manter corpus existente e prosseguir")
        print(f"  2) Recriar corpus (baixar novamente)")
        print(f"  3) Sair")
        
        while True:
            resposta = input("\nEscolha uma opção (1-3): ").strip()
            if resposta == '1':
                print("\n✓ Mantendo corpus existente.")
                return True
            elif resposta == '2':
                print("\nRemovendo partes antigas...")
                for parte in partes_existentes:
                    parte.unlink()
                print("✓ Partes antigas removidas.\n")
                break
            elif resposta == '3':
                print("\n✗ Operação cancelada.")
                return False
            else:
                print("Opção inválida. Digite 1, 2 ou 3.")
    
    print("Baixando corpus Carolina (taxonomia: wik)...")
    print("Isso pode demorar alguns minutos...\n")
    
    documentos = []

    print(f"\n{'='*60}")
    print(f"Carregando taxonomia wik:")
    print(f"{'='*60}")
    
    try:
        print("\nCarregando taxonomia: wik...")
        corpus_wik = load_dataset(
            "carolina-c4ai/corpus-carolina", 
            taxonomy="wik",
            keep_in_memory=False  # Não manter tudo em memória
        )
        corpus_wik = corpus_wik['corpus']
        print(f"✓ wik: {len(corpus_wik)} documentos")
        
    except Exception as e:
        print(f"Erro ao carregar: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\nExtraindo documentos...")
    for item in tqdm(corpus_wik, desc="Processando documentos", unit="doc"):
        texto = item.get('text', '')
        
        if texto and texto.strip():
            documentos.append(texto.strip())
    
    print(f"\n{'='*60}")
    print(f"✓ Total: {len(documentos):,} documentos extraídos")
    print(f"{'='*60}")
    
    if not documentos:
        print("Nenhum documento encontrado!")
        return False
    
    del corpus_wik

    print(f"\nEmbaralhando documentos...")
    random.seed(SEED)
    random.shuffle(documentos)
    
    num_partes = 10
    total_docs = len(documentos)
    tamanho_parte = total_docs // num_partes
    
    print(f"\nDividindo {total_docs:,} documentos em {num_partes} partes...")
    print(f"Aproximadamente {tamanho_parte:,} documentos por parte\n")
    
    for i in tqdm(range(num_partes), desc="Salvando partes", unit="parte"):
        inicio = i * tamanho_parte
        fim = inicio + tamanho_parte if i < num_partes - 1 else total_docs
        
        parte_docs = documentos[inicio:fim]
        arquivo_parte = corpus_dir / f"parte_{i+1:02d}.txt"
        
        with open(arquivo_parte, 'w', encoding='utf-8') as f:
            for doc in parte_docs:
                f.write(doc.replace('\n', ' ').strip() + '\n')
        
        tamanho_mb = arquivo_parte.stat().st_size / (1024 * 1024)
        tqdm.write(f"✓ Parte {i+1:2d}: {len(parte_docs):7,d} documentos ({tamanho_mb:6.2f} MB) -> {arquivo_parte.name}")
    
    print(f"\n{'=' * 60}")
    print(f"✓ Corpus dividido em {num_partes} partes no diretório: {corpus_dir}/")
    print(f"{'=' * 60}")
    
    return True


if __name__ == "__main__":
    preparar_corpus()



