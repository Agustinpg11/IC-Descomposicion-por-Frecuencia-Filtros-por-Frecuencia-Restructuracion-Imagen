CXX      = g++
OPT     ?= -O2
CXXFLAGS = -std=c++17 $(OPT) -Wall
SRC      = wavelet_app.cpp
TARGET   = filtros
IMG      = ref/imagen_ref.jpg
ESCALAS  = 5
RADIO    = 1
FRAMES   = 6

all: $(TARGET)

$(TARGET): $(SRC) stb_image.h stb_image_write.h
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

run: $(TARGET)
	./$(TARGET) $(IMG) $(ESCALAS) 0 $(RADIO) $(FRAMES)

# Compara el checksum global con el guardado en ref/checksum.txt
check: $(TARGET)
	@./$(TARGET) $(IMG) $(ESCALAS) 0 $(RADIO) $(FRAMES) | grep "Checksum" | diff - ref/checksum.txt \
		&& echo "CHECK OK" || (echo "CHECK FALLIDO"; exit 1)

clean:
	rm -f $(TARGET)
	rm -rf capas capas_filtradas resultado_final.png

.PHONY: all run check clean
