# Makefile para TCC-CMTS-IR

ifeq ($(OS),Windows_NT)
    PYTHON = python
    VENV_DIR = .venv
    PIP = .venv\Scripts\pip.exe
    PYTHON_VENV = .venv\Scripts\python.exe
else
    PYTHON = python3
    VENV_DIR = .venv
    PIP = $(VENV_DIR)/bin/pip
    PYTHON_VENV = $(VENV_DIR)/bin/python
endif

.PHONY: help venv install build test clean clean-all run

help:
	@echo "Makefile para TCC-CMTS-IR"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make venv        - Cria ambiente virtual e instala dependências"
	@echo "  make build       - Compila os módulos Cython"
	@echo "  make install     - Instala a biblioteca no ambiente"
	@echo "  make run         - Executa o main.py"
	@echo "  make clean       - Remove arquivos compilados"
	@echo "  make clean-all   - Remove tudo (incluindo venv)"

venv: $(VENV_DIR)/pyvenv.cfg

$(VENV_DIR)/pyvenv.cfg:
	@echo "Criando ambiente virtual..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Atualizando pip e wheel..."
	$(PIP) install --upgrade pip wheel
	@echo "Instalando dependências..."
	$(PIP) install -r requirements.txt
	@echo "Ambiente virtual criado com sucesso!"

build: venv
	@echo "Compilando módulos Cython..."
	$(PYTHON_VENV) setup.py build_ext --inplace
	@echo "Build concluído!"

install:
	@echo "Criando ambiente virtual..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Instalando biblioteca..."
	$(PYTHON_VENV) setup.py install --force
	@echo "Instalação concluída!"

run: build
	@echo "Executando main.py..."
	$(PYTHON_VENV) main.py

clean:
	@echo "Limpando arquivos compilados..."
ifeq ($(OS),Windows_NT)
	-@if exist build rmdir /s /q build
	-@for /r biblioteca %%f in (*.pyd) do @del /f /q "%%f"
	-@for /r biblioteca %%f in (*.so) do @del /f /q "%%f"
else
	-@rm -rf build
	-@find biblioteca -type f -name "*.pyd" -delete
	-@find biblioteca -type f -name "*.so" -delete
endif
	@echo "Limpeza concluída!"

clean-all: clean
	@echo "Removendo ambiente virtual..."
ifeq ($(OS),Windows_NT)
	if exist $(VENV_DIR) rmdir /s /q $(VENV_DIR)
	if exist *.egg-info rmdir /s /q *.egg-info
else
	rm -rf $(VENV_DIR)
	rm -rf *.egg-info
endif
	@echo "Limpeza completa concluída!"
