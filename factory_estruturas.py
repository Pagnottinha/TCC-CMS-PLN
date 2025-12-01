"""Factory para criação e manipulação de estruturas probabilísticas de contagem."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

from biblioteca.estruturas import CMSCU, CML8SCU, CMTS, CL, CML8HSCU

load_dotenv()
SEED = int(os.getenv('SEED', 42))


# Configurações padrão das estruturas
# Baseado em 19.770.871 tokens distintos
CONFIGS_PADRAO = {
    'CMS-CU': {
        'largura': 6_590_290,
        'profundidade': 3,
        'seed': SEED
    },
    'CML8S-CU': {
        'largura': 26_361_160,
        'profundidade': 3,
        'base': 1.08,
        'seed': SEED
    },
    'CMTS-CU': {
        'largura': 389_094,
        'profundidade': 3,
        'base_arvore': 128,
        'seed': SEED
    },
    'CL-CU': {
        'largura': 2_824_410,
        'profundidade': 3,
        'expansao': 4,
        'modo': 0,
        'seed': SEED
    },
    'CL-MU': {
        'largura': 2_824_410,
        'profundidade': 3,
        'expansao': 4,
        'modo': 1,
        'seed': SEED
    },
    'CML8HS-CU': {
        'largura': 6_590_290,
        'profundidade': 3,
        'camadas_lineares': 1,
        'base': 1.08,
        'seed': SEED
    }
}


def criar_estrutura(nome: str, config: Dict[str, Any] = None):
    """
    Cria uma estrutura de dados baseada no nome e configuração.
    
    Se config for None, usa CONFIGS_PADRAO[nome].
    
    Raises:
        ValueError: Se o nome da estrutura não for reconhecido.
        KeyError: Se config for None e nome não existir em CONFIGS_PADRAO.
    """""
    if config is None:
        config = CONFIGS_PADRAO[nome]
    
    if nome == 'CMS-CU':
        return CMSCU(config['largura'], config['profundidade'], config['seed'])
    
    elif nome == 'CML8S-CU':
        return CML8SCU(config['largura'], config['profundidade'], config['base'], config['seed'])
    
    elif nome == 'CMTS-CU':
        return CMTS(config['largura'], config['profundidade'], config['base_arvore'], config['seed'])
    
    elif nome.startswith('CL-'):
        return CL(config['largura'], config['profundidade'], config['expansao'], 
                  config['modo'], config['seed'])
    
    elif nome == 'CML8HS-CU':
        return CML8HSCU(config['largura'], config['profundidade'], config['camadas_lineares'], 
                        config['base'], config['seed'])
    
    else:
        raise ValueError(f"Estrutura desconhecida: {nome}. "
                        f"Estruturas disponíveis: {list(CONFIGS_PADRAO.keys())}")


def criar_todas_estruturas(configs: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cria todas as estruturas disponíveis. Se configs for None, usa CONFIGS_PADRAO."""""
    if configs is None:
        configs = CONFIGS_PADRAO
    
    estruturas = {}
    for nome, config in configs.items():
        estruturas[nome] = criar_estrutura(nome, config)
    
    return estruturas


def obter_classe_estrutura(nome: str):
    """Retorna a classe (não instância) da estrutura para métodos estáticos como carregar()."""""
    mapeamento = {
        'CMS-CU': CMSCU,
        'CML8S-CU': CML8SCU,
        'CMTS-CU': CMTS,
        'CL-CU': CL,
        'CL-MU': CL,
        'CML8HS-CU': CML8HSCU
    }
    
    if nome not in mapeamento:
        raise ValueError(f"Estrutura desconhecida: {nome}. "
                        f"Estruturas disponíveis: {list(mapeamento.keys())}")
    
    return mapeamento[nome]


def carregar_estrutura(nome: str, caminho: str):
    """Carrega uma estrutura salva de arquivo."""""
    classe = obter_classe_estrutura(nome)
    return classe.carregar(caminho)

def incrementar_estrutura(estrutura, elemento: str):
    """Incrementa elemento na estrutura."""""
    if hasattr(estrutura, 'incrementar'):
        estrutura.incrementar(elemento)
    elif hasattr(estrutura, 'add'):
        estrutura.add(elemento)
    else:
        raise AttributeError(f"Estrutura não possui método incrementar/add")

def estimar_estrutura(estrutura, elemento: str) -> int:
    """Estima contagem de elemento na estrutura."""
    if hasattr(estrutura, 'estimar'):
        return estrutura.estimar(elemento)
    elif hasattr(estrutura, 'query'):
        return estrutura.query(elemento)
    else:
        raise AttributeError(f"Estrutura não possui método estimar/query")


# Informações sobre as estruturas
INFO_ESTRUTURAS = {
    'CMS-CU': {
        'nome_completo': 'Count-Min Sketch com Conservative Update',
        'descricao': 'Count-Min Sketch com atualização conservativa',
        'parametros': ['largura', 'profundidade', 'seed']
    },
    'CML8S-CU': {
        'nome_completo': 'Count-Min-Log8 Sketch com Conservative Update',
        'descricao': 'Count-Min Sketch com contadores logarítmicos',
        'parametros': ['largura', 'profundidade', 'base', 'seed']
    },
    'CMTS-CU': {
        'nome_completo': 'Count-Min Tree Sketch com Conservative Update',
        'descricao': 'Count-Min Tree Sketch',
        'parametros': ['largura', 'profundidade', 'base_arvore', 'seed']
    },
    'CL-CU': {
        'nome_completo': 'Count-Less Sketch - Conservative Update',
        'descricao': 'Count-Less Sketch com atualização conservativa',
        'parametros': ['largura', 'profundidade', 'expansao', 'modo', 'seed']
    },
    'CL-MU': {
        'nome_completo': 'Count-Less Sketch - Minimum Update',
        'descricao': 'Count-Less Sketch com atualização mínima',
        'parametros': ['largura', 'profundidade', 'expansao', 'modo', 'seed']
    },
    'CML8HS-CU': {
        'nome_completo': 'Count-Min-Log 8bits Sketch with Conservative Update',
        'descricao': 'CML8SCU com detecção de heavy hitters',
        'parametros': ['largura', 'profundidade', 'camadas_lineares', 'base', 'seed']
    }
}


def listar_estruturas_disponiveis():
    """Lista todas as estruturas disponíveis com suas descrições"""
    print("Estruturas Disponíveis:")
    print("=" * 80)
    for nome, info in INFO_ESTRUTURAS.items():
        config = CONFIGS_PADRAO[nome]
        print(f"\n{nome}")
        print(f"  Nome: {info['nome_completo']}")
        print(f"  Descrição: {info['descricao']}")
        print(f"  Parâmetros: {', '.join(info['parametros'])}")
        print(f"  Config padrão: {config}")


if __name__ == '__main__':
    # Exemplo de uso
    listar_estruturas_disponiveis()