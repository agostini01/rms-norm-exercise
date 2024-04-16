PYSRC=inline_cuda.py
ODIR=prof
GENDIR=gen

# Inline cuda compilation generates source and binary files
# These are the binaries
GEN_OUTPUTS=cuda.cuda.o main.o square_matrix_extension.so .ninja_deps .ninja_log
GEN_OUTPUTS := $(addprefix $(GENDIR)/,$(GEN_OUTPUTS))

run: folders
	python $(PYSRC)

$(GENDIR)/main.o: $(PYSRC)
	python $<

ncu: $(PYSRC) $(GENDIR)/main.o
	ncu --set full -f -o prof/prof python $<

nsys: $(PYSRC) $(GENDIR)/main.o
	nsys profile --stats=true --force-overwrite true -o prof/prof python $<

.PHONY: folders
folders:
	mkdir -p $(ODIR)	
	mkdir -p $(GENDIR)	


.PHONY: clean
clean:
	rm -rf $(ODIR) $(GEN_OUTPUTS)
