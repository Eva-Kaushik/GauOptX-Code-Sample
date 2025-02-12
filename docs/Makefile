# Minimal makefile for Sphinx documentation

# User-configurable variables
SPHINXOPTS    ?= -W  
SPHINXBUILD   ?= sphinx-build
SPHINXPROJ    = GauOptX
SOURCEDIR     = .
BUILDDIR      = _build
NUMJOBS       ?= auto  # Enables parallel builds

CHECK_SPHINX := $(shell command -v $(SPHINXBUILD) 2> /dev/null)

ifndef CHECK_SPHINX
$(error "Sphinx is not installed. Install it with: pip install sphinx")
endif

# Default target: show help message
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html latexpdf clean

# Common build targets
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -j $(NUMJOBS) $(SPHINXOPTS)

latexpdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" -j $(NUMJOBS) $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"
	@echo "Cleaned build directory."

# Catch-all target: Forward unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -j $(NUMJOBS) $(SPHINXOPTS) $(O)