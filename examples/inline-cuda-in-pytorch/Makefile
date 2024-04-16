PYSRC=inline_cuda.py
ODIR=prof
GENDIR=gen

# Inline cuda compilation generates source and binary files
# These are the binaries
GEN_OUTPUTS=cuda.cuda.o main.o square_matrix_extension.so .ninja_deps .ninja_log
GEN_OUTPUTS := $(addprefix $(GENDIR)/,$(GEN_OUTPUTS))

run: folders
	python3 $(PYSRC)

.PHONY: folders
folders:
	mkdir -p $(ODIR)	
	mkdir -p $(GENDIR)	


.PHONY: clean
clean:
	rm -rf $(ODIR) $(GEN_OUTPUTS)
