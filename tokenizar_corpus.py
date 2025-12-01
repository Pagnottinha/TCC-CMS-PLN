"""Tokenização e normalização do corpus Carolina."""
from pathlib import Path
import re
from simplemma import lemmatize
import time
import unicodedata
import spacy
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.symbols import ORTH

nlp = spacy.blank("pt")
nlp.max_length = 2000000

_REGEX_URLS = re.compile(
    r'(?:'
    r'https?://[^\s]+|'
    r'ftp://[^\s]+|'
    r'www\.[a-zA-Z0-9\-]+\.[^\s]+|'
    r'doi:\s*10\.\S+|'
    r'\b10\.\d{4,}/[^\s]+|'
    r'\bisbn[\-:\s]+(?:97[89][\-\s]?)?(?:\d[\-\s]?){9,12}\d\b|'
    r'\b97[89][\-\s](?:\d[\-\s]?){9}\d\b|'
    r'\b\d[\-\s](?:\d[\-\s]){7,8}[\dXx]\b|'
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|'
    r'@\w+|'
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    r')',
    re.IGNORECASE
)
_REGEX_PONTUACOES_CONSECUTIVAS = re.compile(r'[\.\-;:,!?]{2,}')
_REGEX_ABREV_ORDINAL = re.compile(r'^([a-záàâãéêíóôõúçü]+)\.?\s*[ºª]$', re.IGNORECASE)
_REGEX_NAO_PT = re.compile(r'[^a-záàâãéêíóôõúçüA-ZÁÀÂÃÉÊÍÓÔÕÚÇÜµ\s\-–—\d_.,;:!?²³ªº/°%$€£¥₹₽¢()[\]{}]')
_REGEX_SEP_PARENTESES_ESQ = re.compile(r'([^\s])([\(\[\{])')
_REGEX_SEP_PARENTESES_DIR = re.compile(r'([\)\]\}])([^\s])')
_REGEX_SEP_PONT_LETRA = re.compile(r'([a-zA-Zºª°])([,;:])')
_REGEX_SEP_PONT_NUM = re.compile(r'(\d)([,;:])(?=[a-záàâãéêíóôõúçüA-ZÁÀÂÃÉÊÍÓÔÕÚÇÜ])')

_REGEX_CLITICOS = re.compile(
    r'(-('
    r'me|te|se|o|a|lhe|nos|vos|os|as|lhes|'
    r'lo|la|los|las|no|na|nas|'
    r'ei|ás|á|à|emos|eis|ão|'
    r'ia|ias|iamos|íeis|iam'
    r'))+$',
    re.IGNORECASE
)
_REGEX_ABREVIACAO_NUMERO = re.compile(r'^[nN]\.?\s*[º°]\.?$')
_REGEX_EMOTICONS = re.compile(
    r'(?<!\S)'
    r'(?:'
    r'(?::|;|=)(?:-)?(?:\)|D|\(|P|p|\||\\|/|\*|\$|@|3|>|<|\[|\]|\{|\})'
    r'|(?:\)|D|\(|P|p|\||\\|/|\*|\$|@|3|>|<|\[|\]|\{|\})(?:-)?(?::|;|=)'
    r'|<3'
    r'|[xX][Dd]'
    r'|\^_\^|\^\.?\^'
    r'|>_<|>.<'
    r'|-_-|\._\.'
    r'|[oO]_[oO]|0_0'
    r'|[Tt]_[Tt]|[Tt][Tt]|[Tt]o[Tt]'
    r')'
    r'(?!\S)',
    re.IGNORECASE
)
_REGEX_MULTIPLOS_ESPAÇOS = re.compile(r'\s+')

_ABREVIACOES_COMUNS = {
    'dr': 'doutor', 'dra': 'doutora', 'sr': 'senhor', 'sra': 'senhora', 'srª': 'senhora',
    'prof': 'professor', 'profº': 'professor', 'profa': 'professora', 'profª': 'professora',
    'jr': 'júnior', 'srta': 'senhorita',
    'ltda': 'limitada', 'cia': 'companhia', 's.a': 'sociedade_anônima',
    'etc': 'etcétera', 'etcs': 'etcéteras', 'ex': 'exemplo',
    'tel': 'telefone', 'fax': 'fax', 'cel': 'celular',
    'av': 'avenida', 'r': 'rua', 'rod': 'rodovia',
    'vol': 'volume', 'ed': 'edição', 'pg': 'página', 'pag': 'página', 'pág': 'página',
    'cap': 'capítulo', 'art': 'artigo', 'obs': 'observação',
    'gov': 'governo', 'adm': 'administração',
    'dep': 'departamento', 'sec': 'secretaria', 'pref': 'prefeitura',
    'arq': 'arquiteto', 'eng': 'engenheiro', 'med': 'médico',
    'cons': 'conselheiro'
}

_UNIDADES_EXPANDIDAS = {
    'km': 'quilometro', 'm': 'metro', 'cm': 'centimetro', 'mm': 'milimetro',
    'ft': 'pe', 'in': 'polegada', 'yd': 'jarda', 'mi': 'milha',
    'nm': 'nanometro', 'μm': 'micrometro',
    'pc': 'parsec', 'ua': 'unidade_astronomica', 'al': 'ano_luz',
    'kg': 'quilograma', 'g': 'grama', 'mg': 'miligrama',
    'lb': 'libra', 'oz': 'onca',
    'l': 'litro', 'ml': 'mililitro', 'dl': 'decilitro', 'cl': 'centilitro',
    'gal': 'galao', 'qt': 'quarto',
    'h': 'hora', 'min': 'minuto', 's': 'segundos', 'ms': 'milissegundo',
    's²': 'segundo_quadrado',
    'ha': 'hectare',
    'km2': 'quilometro_quadrado', 'km²': 'quilometro_quadrado', 
    'm2': 'metro_quadrado', 'm²': 'metro_quadrado',
    'cm2': 'centimetro_quadrado', 'cm²': 'centimetro_quadrado',
    'mm2': 'milimetro_quadrado', 'mm²': 'milimetro_quadrado',
    'km3': 'quilometro_cubico', 'km³': 'quilometro_cubico', 
    'm3': 'metro_cubico', 'm³': 'metro_cubico',
    'cm3': 'centimetro_cubico', 'cm³': 'centimetro_cubico', 
    'mm3': 'milimetro_cubico', 'mm³': 'milimetro_cubico',
    'kmh': 'quilometro_hora', 'mph': 'milha_hora',
    'hpa': 'hectopascal', 'pa': 'pascal', 'kpa': 'quilopascal',
    'tb': 'terabyte', 'gb': 'gigabyte', 'mb': 'megabyte', 'kb': 'quilobyte'
}

_TERMOS_PROTEGIDOS = {
    'quadrado', 'quadrados',
    'cúbico', 'cúbicos', 'cubico', 'cubicos'
}


prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
infix_re = compile_infix_regex(nlp.Defaults.infixes)

nlp.tokenizer = Tokenizer(
    nlp.vocab,
    rules={},
    prefix_search=prefix_re.search,
    suffix_search=suffix_re.search,
    infix_finditer=infix_re.finditer
)

SIGLAS_ESPECIAIS = {
    "Ph.D.": [{ORTH: "Ph.D."}],
    "ph.d.": [{ORTH: "ph.d."}],
    "Ph.D": [{ORTH: "Ph.D"}],
    "ph.d": [{ORTH: "ph.d"}],
    "M.Sc.": [{ORTH: "M.Sc."}],
    "m.sc.": [{ORTH: "m.sc."}],
    "B.Sc.": [{ORTH: "B.Sc."}],
    "b.sc.": [{ORTH: "b.sc."}]
}

for sigla, attrs in SIGLAS_ESPECIAIS.items():
    nlp.tokenizer.add_special_case(sigla, attrs)

@Language.component("merge_abreviacao_numero")
def merge_abreviacao_numero(doc: Doc):
    """Agrupa 'n.º' antes de dígitos."""
    with doc.retokenize() as retok:
        i = 0
        while i < len(doc):
            if (
                i + 2 < len(doc)
                and doc[i].text == "n"
                and doc[i+1].text == "."
                and doc[i+2].text in {"º", "°"}
            ):
                retok.merge(doc[i:i+3])
                i += 1
                continue
            
            if (
                i + 2 < len(doc)
                and doc[i].text == "n"
                and doc[i+1].text == "."
                and doc[i+2].text.isdigit()
            ):
                retok.merge(doc[i:i+2])
                i += 1
                continue
            
            i += 1
    return doc

@Language.component("normalizador_customizado")
def normalizador_customizado(doc: Doc):
    """Normaliza tokens, remove stopwords e lematiza."""
    SIMBOLOS_UTEIS = {'%', '$', '€', '£', '¥', '°', 'º', 'ª'}
    
    for token in doc:
        if token.text in SIMBOLOS_UTEIS:
            token.lemma_ = token.text.lower()
            continue

        if token.is_punct or token.is_space:
            token.lemma_ = ""
            continue
            
        texto_normalizado = normalizar_token(token.text) 
        
        if not texto_normalizado or texto_normalizado in STOP_WORDS:
            token.lemma_ = ""
            continue
            
        if texto_normalizado in _UNIDADES_EXPANDIDAS.values():
            token.lemma_ = texto_normalizado
            continue
             
        if texto_normalizado in _TERMOS_PROTEGIDOS:
            token.lemma_ = texto_normalizado
            continue
        
        lema = lemmatize(texto_normalizado, lang='pt')
        
        if lema in STOP_WORDS:
            token.lemma_ = ""
        else:
            token.lemma_ = lema
            
    return doc

nlp.add_pipe("merge_abreviacao_numero", first=True)
nlp.add_pipe("normalizador_customizado", last=True)

def limpar_texto(texto: str) -> str:
    """Remove ruídos e normaliza o texto para tokenização."""
    texto = unicodedata.normalize('NFC', texto)
    
    if any(c in texto for c in ['@', '.', '/', ':']):
        texto = _REGEX_URLS.sub(' ', texto)
    
    texto = _REGEX_EMOTICONS.sub(' ', texto)
    
    texto = _REGEX_SEP_PARENTESES_ESQ.sub(r'\1 \2', texto)
    texto = _REGEX_SEP_PARENTESES_DIR.sub(r'\1 \2', texto)
    
    texto = _REGEX_SEP_PONT_LETRA.sub(r'\1 \2 ', texto)
    texto = _REGEX_SEP_PONT_NUM.sub(r'\1 \2 ', texto)
    
    texto = _REGEX_PONTUACOES_CONSECUTIVAS.sub(' ', texto)
    
    texto = _REGEX_NAO_PT.sub(' ', texto)
    
    texto = _REGEX_MULTIPLOS_ESPAÇOS.sub(' ', texto).strip()
    
    return texto

_VERBOS_CLITICOS_MAP = {
    'fê': 'fazer',
    'vê': 'ver',
    'revertê': 'reverter'
}

def normalizar_token(token: str) -> str:
    """Normaliza um token aplicando regras semânticas (abreviações, siglas, unidades)."""
    token_lower = token.strip().lower()
    token_core = token_lower.rstrip('.,;:!?')
    
    if _REGEX_ABREVIACAO_NUMERO.match(token_lower):
        return 'numero'

    match_abrev_ordinal = _REGEX_ABREV_ORDINAL.match(token_lower)
    if match_abrev_ordinal:
        abrev = match_abrev_ordinal.group(1)
        if abrev in _ABREVIACOES_COMUNS:
            return _ABREVIACOES_COMUNS[abrev].lower()
    
    if token_core in _ABREVIACOES_COMUNS:
        return _ABREVIACOES_COMUNS[token_core].lower()

    if (
        token_core.count('.') >= 1
        and any(c.isalpha() for c in token_core)
        and not any(c.isdigit() for c in token_core)
    ):
        return token_lower.replace('.', '')

    if token_core in _UNIDADES_EXPANDIDAS:
        return _UNIDADES_EXPANDIDAS[token_core]

    if '-' in token_core:
        sem_clitico = _REGEX_CLITICOS.sub('', token_core)
        if sem_clitico != token_core:
            return _VERBOS_CLITICOS_MAP.get(sem_clitico, sem_clitico)
    
    return token_core.lower()

def tokenizar_corpus():
    """Orquestra o processo completo de tokenização do corpus."""
    
    print("=" * 70)
    print("TOKENIZAÇÃO DO CORPUS CAROLINA")
    print("=" * 70)
    
    corpus_dir = Path("corpus")
    tokenizado_dir = corpus_dir / "tokenizado"
    tokenizado_dir.mkdir(exist_ok=True)
    
    partes_originais = sorted(list(corpus_dir.glob("parte_*.txt")))
    if not partes_originais:
        print("\nErro: Corpus original não encontrado!")
        print("   Esperado: corpus/parte_01.txt, parte_02.txt, ...")
        print("   Execute primeiro o script corpus.py\n")
        return False
    
    print(f"\n✓ Encontradas {len(partes_originais)} partes do corpus original.")
    
    partes_tokenizadas = sorted(list(tokenizado_dir.glob("parte_*.txt")))
    if partes_tokenizadas:
        print(f"\n✓ Encontradas {len(partes_tokenizadas)} partes já tokenizadas:")
        for parte in partes_tokenizadas:
            linhas = sum(1 for _ in open(parte, encoding='utf-8'))
            tamanho_mb = parte.stat().st_size / (1024 * 1024)
            print(f"  - {parte.name}: {linhas:6d} documentos ({tamanho_mb:.2f} MB)")
        
        print(f"\nO que deseja fazer?")
        print(f"  1) Manter corpus tokenizado existente e prosseguir")
        print(f"  2) Retokenizar corpus (processar novamente)")
        print(f"  3) Sair")
        
        while True:
            resposta = input("\nEscolha uma opção (1-3): ").strip()
            if resposta == '1':
                print("\n✓ Mantendo corpus tokenizado existente.")
                return True
            elif resposta == '2':
                print("\nRemovendo partes tokenizadas antigas...")
                for parte in partes_tokenizadas:
                    parte.unlink()
                print("✓ Partes antigas removidas.\n")
                break
            elif resposta == '3':
                print("\n✗ Operação cancelada.")
                return False
            else:
                print("Opção inválida. Digite 1, 2 ou 3.")
    
    print("\nPipeline de processamento:")
    print("   1. Limpeza de ruído")
    print("   2. Tokenização e Normalização")
    print("   3. Lematização e filtragem de stopwords")
    
    print(f"\n{'=' * 70}")
    print(f"Iniciando tokenização de {len(partes_originais)} partes...")
    print(f"{'=' * 70}\n")
    
    tempo_total_inicio = time.time()
    estatisticas = {
        'total_documentos_entrada': 0,
        'total_documentos_saida': 0,
        'total_tokens': 0,
        'documentos_vazios': 0
    }
    
    for idx, arquivo_original in enumerate(partes_originais, 1):
        tempo_parte_inicio = time.time()
        
        print(f"[{idx:02d}/{len(partes_originais)}] {arquivo_original.name}")
        
        with open(arquivo_original, 'r', encoding='utf-8') as f:
            documentos = [linha.strip() for linha in f if linha.strip()]
        
        estatisticas['total_documentos_entrada'] += len(documentos)
        print(f"       Entrada: {len(documentos):,} documentos")
        
        arquivo_tokenizado = tokenizado_dir / arquivo_original.name
        vazios = 0
        textos_limpos_gerador = (limpar_texto(doc) for doc in documentos)
        documentos_tokenizados = []
        tokens_processados = 0

        docs_spacy = nlp.pipe(textos_limpos_gerador, batch_size=2000, n_process=1)

        print("       Processando documentos...")
        for doc in docs_spacy:
            tokens_finais = [t.lemma_ for t in doc if t.lemma_]

            if tokens_finais:
                documentos_tokenizados.append(';'.join(tokens_finais))
                tokens_processados += len(tokens_finais)
            else:
                vazios += 1
                print(f"         • Documento vazio encontrado na linha {len(documentos_tokenizados) + vazios}.")
                      
        print(f"       Salvando parte tokenizada...")
        with open(arquivo_tokenizado, 'w', encoding='utf-8') as f:
            for doc in documentos_tokenizados:
                f.write(doc + '\n')
        
        tempo_parte = time.time() - tempo_parte_inicio
        tamanho_mb = arquivo_tokenizado.stat().st_size / (1024 * 1024)
        velocidade = tokens_processados / tempo_parte if tempo_parte > 0 else 0
        
        estatisticas['total_documentos_saida'] += len(documentos_tokenizados)
        estatisticas['total_tokens'] += tokens_processados
        estatisticas['documentos_vazios'] += vazios
        
        print(f"       Saída: {len(documentos_tokenizados):,} documentos ({tamanho_mb:.2f} MB)")
        print(f"       Tokens: {tokens_processados:,} ({velocidade:,.0f} tokens/seg)")
        print(f"       Tempo: {tempo_parte:.1f}s")
        if vazios > 0:
            print(f"       Vazios: {vazios:,} documentos sem tokens válidos")
        print()
    
    tempo_total = time.time() - tempo_total_inicio
    velocidade_media = estatisticas['total_tokens'] / tempo_total if tempo_total > 0 else 0
    reducao = (1 - estatisticas['total_documentos_saida'] / estatisticas['total_documentos_entrada']) * 100
    
    print(f"{'=' * 70}")
    print(f"TOKENIZAÇÃO CONCLUÍDA COM SUCESSO")
    print(f"{'=' * 70}")
    print(f"\nEstatísticas Finais:")
    print(f"   • Documentos entrada:  {estatisticas['total_documentos_entrada']:>10,}")
    print(f"   • Documentos saída:    {estatisticas['total_documentos_saida']:>10,}")
    print(f"   • Documentos vazios:   {estatisticas['documentos_vazios']:>10,} ({reducao:.1f}% redução)")
    print(f"   • Total de tokens:     {estatisticas['total_tokens']:>10,}")
    print(f"   • Velocidade média:    {velocidade_media:>10,.0f} tokens/seg")
    print(f"   • Tempo total:         {tempo_total/60:>10.1f} minutos")
    print(f"\nCorpus tokenizado salvo em: {tokenizado_dir.absolute()}/")
    print(f"{'=' * 70}\n")
    
    return True


if __name__ == "__main__":
    sucesso = tokenizar_corpus()
    
    if sucesso:
        print("Processo concluído!")
    else:
        print("Erro na tokenização. Verifique os arquivos.")
    # exemplos = [
    #     # Casos bíblicos típicos
    #     "Gênesis 18:23-33;Êxodo 1:17, 2:11-14",
    #     "Mateus 5:3-10;João 3:16",
    #     "Romanos 8:28;Salmos 23:1-4",
    #     "1 Coríntios 13:4-7;Gálatas 5:22-23",
    #     "Apocalipse 3:20;Lucas 15:11-32",

    #     # Variações com espaços
    #     "Gênesis 18:23-33 ; Êxodo 1:17, 2:11-14",
    #     "Mateus 5:3-10 ; João 3:16",
    #     "Romanos 8:28 ; Salmos 23:1-4",

    #     # Casos que NÃO deveriam acionar a regex (número;número)
    #     "Lista de exercícios 3;4, 5;6 e 7;8",
    #     "Faixa etária 18;25 anos",

    #     # Casos mistos número + símbolo/letra
    #     "Código interno 12;A e 13;B",
    #     "Taxa 3;% ao mês",
    #     "Nota 10;🙂 no relatório",

    #     # Casos com letras acentuadas (onde você quer separar)
    #     "Capítulo 3;Êxodo e 4;Gênesis",
    #     "Referência 2;Árvore e 3;Óleo",

    #     # Casos com hífen e ponto-e-vírgula grudados
    #     "Seção 10-12;Introdução",
    #     "Parágrafo 5-7;Conclusão geral"
    # ]

    # for exemplo in exemplos:
    #     print(limpar_texto(exemplo))
