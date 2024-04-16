PYSRC=triton_square.py
ODIR=prof

run: folders
	python3 $(PYSRC) 2>&1 | tee $(ODIR)/square.log

.PHONY: folders
folders:
	mkdir -p $(ODIR)	

.PHONY: clean
clean:
	rm -rf $(ODIR) results.html performance_of_square_function.csv