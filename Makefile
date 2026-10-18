# define variables
PYTHON = venv/bin/python
PYINSTALLER_VENV := venv_no_cuda
PYINSTALLER := $(PYINSTALLER_VENV)/bin/pyinstaller
CurrentDir := $(shell pwd)
BUILD_DIR = dist
BUILD_CACHE = handtex.egg-info build
DIR_ICONS := icons
UI_DIR := ui_files
UI_OUTPUT_DIR := handtex/ui_generated_files
UIC_COMPILER := venv/bin/pyside6-uic
BLACK_LINE_LENGTH := 100
BLACK_TARGET_DIR := handtex/
BLACK_EXCLUDE_PATTERN := "^$(UI_OUTPUT_DIR)/.*"

# default target
fresh-install: clean build install

refresh-assets: build-icon-cache compile-ui bundle-symbols

run:
    # PYTHONPATH is needed for the local imports to work.
	PYTHONPATH=. $(PYTHON) handtex/main.py

run-pyinstaller-venv:
	PYTHONPATH=. $(PYINSTALLER_VENV)/bin/python handtex/main.py

setup-venv:
	python -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

setup-venv-training:
	python -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements_training.txt

# build target
build:
	$(PYTHON) -m build --outdir $(BUILD_DIR)

# install target
install:
	$(PYTHON) -m pip install $(BUILD_DIR)/*.whl

# clean target
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BUILD_CACHE)
	rm -rf AUR/handtex/pkg
	rm -rf AUR/handtex/src
	rm -rf AUR/handtex/*.tar.gz
	rm -rf AUR/handtex/*.tar.zst

release: confirm
	$(PYTHON) -m twine upload $(BUILD_DIR)/*

bundle-symbols:
	tar --transform='s:.*/::' --owner=0 --group=0 --mode=644 --mtime='1970-01-01' -cJf handtex/data/symbols.tar.xz symbols/svg/*.svg

# compile .ui files
compile-ui:
	for file in $(UI_DIR)/*.ui; do \
		basename=`basename $$file .ui`; \
		$(UIC_COMPILER) $$file -o $(UI_OUTPUT_DIR)/ui_$$basename.py; \
	done

build-icon-cache:
	$(PYTHON) $(DIR_ICONS)/build_icon_cache.py
	$(PYTHON) $(DIR_ICONS)/copy_from_dark_to_light.py
# format the code
black-format:
	find $(BLACK_TARGET_DIR) -type f -name '*.py' | grep -Ev $(BLACK_EXCLUDE_PATTERN) | xargs black --line-length $(BLACK_LINE_LENGTH)

confirm:
	@read -p "Are you sure you want to proceed? (yes/no): " CONFIRM; \
	if [ "$$CONFIRM" = "yes" ]; then \
		echo "Proceeding..."; \
	else \
		echo "Aborted by user."; \
		exit 1; \
	fi

build-elf:
	# Use torch from: https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp313-cp313-linux_x86_64.whl
	# Use torchvision from: https://download.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp313-cp313-linux_x86_64.whl
	$(PYINSTALLER_VENV)/bin/python -m PyInstaller handtex/main.py \
		--onedir --noconfirm --clean --workpath=build --distpath=dist-elf --windowed \
		--name="HandTeX" \
		--copy-metadata=numpy \
		--copy-metadata=packaging \
		--copy-metadata=pillow \
		--copy-metadata=torch

	# This stupid thing refuses to collect data, so do it manually:
	@echo "Copying data files..."
	mkdir -p dist-elf/HandTeX/_internal/handtex
	cp -r handtex/data dist-elf/HandTeX/_internal/handtex/
	@echo "Purging __pycache__ directories..."
	@find dist-elf/HandTeX/_internal/handtex -type d -name "__pycache__"
	@find dist-elf/HandTeX/_internal/handtex -type d -name "__pycache__" -exec rm -rf {} \; || true

#	@echo "Purging CUDA related files from _internal directory..."
#	@find dist-elf/HandTeX/_internal -type f \( \
#		-name 'libtorch_cuda.so' -o \
#		-name 'libc10_cuda.so' -o \
#		-name 'libcusparse.so*' -o \
#		-name 'libcurand.so*' -o \
#		-name 'libcudnn.so*' -o \
#		-name 'libcublasLt.so*' -o \
#		-name 'libcublas.so*' -o \
#		-name 'libcupti.so*' -o \
#		-name 'libcufft.so*' -o \
#		-name 'libcudart.so*' -o \
#		-name 'libnv*' -o \
#		-name 'libnccl.so*' \
#		\) -exec rm -rf {} \;



.PHONY: confirm clean build install fresh-install release black-format compile-ui build-icon-cache refresh-assets