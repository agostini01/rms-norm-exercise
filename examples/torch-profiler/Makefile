TARGET=gemm

run: folders
	python3 test_pt_profiler.py

all: gemm

$(TARGET): $(TARGET).cu
	nvcc $< -o $@ 

ncu: $(TARGET) folders
	ncu --set full -f -o prof/$< $<

nsys: $(TARGET) folders
	nsys profile --stats=true --force-overwrite true -o prof/$< $<

.PHONY: folders
folders:
	mkdir -p prof	

.PHONY: clean
clean:
	rm -rf $(TARGET) prof