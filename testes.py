"""
Testes das estruturas de dados probabilísticas
"""
from biblioteca.estruturas import CMSCU, CMLS8CU, CMLS8CUH, CL, CMTS
from collections import Counter
import random
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração global
SEED = int(os.getenv('SEED', 42))

# Configuração das estruturas para teste
ESTRUTURAS = [
    {
        'nome': 'CMS_CU',
        'classe': CMSCU,
        'params': {'largura': 1000, 'profundidade': 4, 'seed': SEED},
        'params_grande': {'largura': 333333, 'profundidade': 3, 'seed': SEED},
        'formato_saida': lambda x: str(x),
    },
    {
        'nome': 'CMLS8CU',
        'classe': CMLS8CU,
        'params': {'largura': 3000, 'profundidade': 4, 'base': 1.08, 'seed': SEED},
        'params_grande': {'largura': 1333333, 'profundidade': 3, 'base': 1.08, 'seed': SEED},
        'formato_saida': lambda x: f"{x:.2f}",
    },
    {
        'nome': 'CMLS8CUH',
        'classe': CMLS8CUH,
        'params': {'largura': 3000, 'profundidade': 4, 'camadas_lineares': 2, 'base': 1.08, 'seed': SEED},
        'params_grande': {'largura': 250000, 'profundidade': 4, 'camadas_lineares': 1, 'base': 1.08, 'seed': SEED},
        'formato_saida': lambda x: f"{x:.2f}",
    },
    {
        'nome': 'CL (Conservative Update)',
        'classe': CL,
        'params': {'largura': 500, 'profundidade': 4, 'expansao': 4, 'modo': 0, 'seed': SEED},
        'params_grande': {'largura': 142857, 'profundidade': 3, 'expansao': 4, 'modo': 0, 'seed': SEED},
        'formato_saida': lambda x: str(x),
    },
    {
        'nome': 'CL (Minimal Update)',
        'classe': CL,
        'params': {'largura': 500, 'profundidade': 4, 'expansao': 4, 'modo': 1, 'seed': SEED},
        'params_grande': {'largura': 142857, 'profundidade': 3, 'expansao': 4, 'modo': 1, 'seed': SEED},
        'formato_saida': lambda x: str(x),
    },
    {
        'nome': 'CMTS',
        'classe': CMTS,
        'params': {'largura': 250, 'profundidade': 4, 'base_arvore': 128, 'seed': SEED},
        'params_grande': {'largura': 19680, 'profundidade': 3, 'base_arvore': 128, 'seed': SEED},
        'formato_saida': lambda x: str(x),
    }
]


def test_basic_operations():
    """Teste básico de operações"""
    print("=" * 60)
    print("TESTE 1: Operações Básicas")
    print("=" * 60)
    
    sketches = []
    for config in ESTRUTURAS:
        sketch = config['classe'](**config['params'])
        sketches.append((config['nome'], sketch, config['formato_saida']))
        print(f"✓ {config['nome']} criado: {sketch}")
        print(f"  Tamanho em bytes: {sketch.sizeof():,}")
    
    print("\nAdicionando elementos...")
    elementos = ["hello", "world", "hello", "hello", "python"]
    for elem in elementos:
        for _, sketch, _ in sketches:
            sketch.incrementar(elem)
    
    print("\nFrequências:")
    for nome, sketch, fmt in sketches:
        print(f"\n{nome}:")
        for elem in ["hello", "world", "python", "notfound"]:
            freq = sketch.estimar(elem)
            print(f"  '{elem}': {fmt(freq)}")
    print()


def test_multiple_additions():
    """Teste com múltiplas adições do mesmo elemento"""
    print("=" * 60)
    print("TESTE 2: Múltiplas Adições")
    print("=" * 60)
    
    element = "test_element"
    num_additions = 100
    
    print(f"Adicionando '{element}' {num_additions} vezes...\n")
    
    for config in ESTRUTURAS:
        params = config['params'].copy()
        params['largura'] = 500
        params['profundidade'] = 3
        if 'base' not in params:
            params['seed'] = 123
        else:
            params['seed'] = 123
            params['largura'] = 1500
        
        sketch = config['classe'](**params)
        
        for _ in range(num_additions):
            sketch.incrementar(element)
        
        freq = sketch.estimar(element)
        precisao = (freq / num_additions) * 100
        
        print(f"{config['nome']}:")
        print(f"  Frequência estimada: {config['formato_saida'](freq)}")
        print(f"  Frequência real: {num_additions}")
        print(f"  Precisão: {precisao:.2f}%\n")


def test_create_from_expected_error():
    """Teste criação com parâmetros de erro esperado"""
    print("=" * 60)
    print("TESTE 3: Criação com Parâmetros de Erro")
    print("=" * 60)
    
    deviation = 0.001
    error = 0.01
    
    for config in ESTRUTURAS:
        if hasattr(config['classe'], 'criar_apartir_erro_esperado'):
            if 'base' in config['params']:
                sketch = config['classe'].criar_apartir_erro_esperado(deviation, error, config['params']['base'], SEED)
            else:
                sketch = config['classe'].criar_apartir_erro_esperado(deviation, error, SEED)
            
            print(f"✓ {config['nome']} criado com deviation={deviation}, error={error}")
            print(f"  Sketch: {sketch}")
            print(f"  Tamanho: {sketch.sizeof():,} bytes\n")


def test_frequency_estimation():
    """Teste de estimativa de frequência com diferentes elementos"""
    print("=" * 60)
    print("TESTE 4: Estimativa de Frequência de Vários Elementos")
    print("=" * 60)
    
    data = ["apple", "banana", "apple", "cherry", "apple", 
            "banana", "apple", "date", "apple", "banana"]
    
    print("Stream de dados:", data)
    print("\nAdicionando elementos aos sketches...")
    
    sketches = []
    for config in ESTRUTURAS:
        params = config['params'].copy()
        params['largura'] = 2000
        params['seed'] = 50
        if 'base' in params:
            params['largura'] = 6000
        
        sketch = config['classe'](**params)
        for item in data:
            sketch.incrementar(item)
        
        sketches.append((config['nome'], sketch, config['formato_saida']))
    
    real_freq = Counter(data)
    
    for nome, sketch, fmt in sketches:
        print(f"\n{nome} - Comparando frequências:")
        print(f"{'Elemento':<15} {'Real':<10} {'Estimada':<12} {'Diferença':<10}")
        print("-" * 50)
        
        for item in real_freq:
            estimated = sketch.estimar(item)
            real = real_freq[item]
            diff = estimated - real
            print(f"{item:<15} {real:<10} {fmt(estimated):<12} {fmt(diff):+<10}")
        
        not_in_data = "elderberry"
        print(f"\n✓ Elemento não existente '{not_in_data}': {fmt(sketch.estimar(not_in_data))}")
    print()


def test_large_dataset():
    """Teste com dataset maior"""
    print("=" * 60)
    print("TESTE 5: Dataset Grande")
    print("=" * 60)
    
    quantidade = 1000000
    random.seed(SEED)
    
    # Criar sketches
    sketches = []
    print("Estruturas:")
    for config in ESTRUTURAS:
        sketch = config['classe'](**config['params_grande'])
        sketches.append((config['nome'], sketch, config['formato_saida']))
        print(f"  {config['nome']:<10} - Tamanho: {sketch.sizeof():>12,} bytes")
    
    if len(sketches) > 1:
        ratio = sketches[1][1].sizeof() / sketches[0][1].sizeof()
        print(f"\nEconomia de memória: {(1 - ratio)*100:+.1f}%")
    print()
    
    # Gerar dados
    elements = []
    for i in range(quantidade):
        elements.extend([f"item_{i}"] * (quantidade // (quantidade - i)))
    
    print(f"Adicionando {len(elements):,} elementos aos sketches...")
    for elem in elements:
        for _, sketch, _ in sketches:
            sketch.incrementar(elem)
    
    print(f"✓ Total de elementos processados: {len(elements):,}\n")
    
    # Verificar frequências
    step = quantidade // 10
    
    for nome, sketch, fmt in sketches:
        print(f"{nome} - Verificando frequências:")
        are = 0
        for i in range(quantidade):
            expected = quantidade // (quantidade - i)
            estimated = sketch.estimar(f"item_{i}")
            error_rate = abs(estimated - expected) / expected
            are += error_rate
            if (i % step == 0 or i == quantidade - 1):
                print(f"  item_{i}: esperado={expected}, estimado={fmt(estimated)}, erro={error_rate*100:.2f}%")
        
        print(f"ARE ({nome}): {are / quantidade:.6f}\n")


def main():
    """Função principal para executar todos os testes"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "TESTES DE ESTRUTURAS PROBABILÍSTICAS" + " " * 10 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        test_basic_operations()
        test_multiple_additions()
        test_create_from_expected_error()
        test_frequency_estimation()
        test_large_dataset()
        
        print("=" * 60)
        print("✓ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
