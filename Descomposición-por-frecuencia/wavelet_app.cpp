#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <filesystem>
namespace fs = std::filesystem;

// Reflejo en bordes para evitar artefactos en los límites de la imagen
inline int clamp_coord(int pos, int max_size) {
    if (pos < 0) return -pos;
    if (pos >= max_size) return 2 * max_size - pos - 2;
    return pos;
}

// Blur a trous
// Convolución separable a trous (B-spline cúbico: 1/16, 4/16, 6/16, 4/16, 1/16)
// Hace el blur en dos pasos de la imagen
void atrous_blur(const std::vector<float>& src, std::vector<float>& dst, 
                 int w, int h, int channels, int scale) {
    int step = 1 << scale; // 2^scale (1, 2, 4, 8, ...)
    std::vector<float> temp(src.size());

    // 1. Paso horizontal
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                int x_m2 = clamp_coord(x - 2 * step, w);
                int x_m1 = clamp_coord(x - 1 * step, w);
                int x_p1 = clamp_coord(x + 1 * step, w);
                int x_p2 = clamp_coord(x + 2 * step, w);

                float val = 0.375f * src[(y * w + x) * channels + c]
                          + 0.25f  * (src[(y * w + x_m1) * channels + c] + src[(y * w + x_p1) * channels + c])
                          + 0.0625f * (src[(y * w + x_m2) * channels + c] + src[(y * w + x_p2) * channels + c]);

                temp[(y * w + x) * channels + c] = val;
            }
        }
    }

    // 2. Paso vertical
    for (int y = 0; y < h; ++y) {
        int y_m2 = clamp_coord(y - 2 * step, h);
        int y_m1 = clamp_coord(y - 1 * step, h);
        int y_p1 = clamp_coord(y + 1 * step, h);
        int y_p2 = clamp_coord(y + 2 * step, h);

        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                float val = 0.375f * temp[(y * w + x) * channels + c]
                          + 0.25f  * (temp[(y_m1 * w + x) * channels + c] + temp[(y_p1 * w + x) * channels + c])
                          + 0.0625f * (temp[(y_m2 * w + x) * channels + c] + temp[(y_p2 * w + x) * channels + c]);

                dst[(y * w + x) * channels + c] = val;
            }
        }
    }
}

// Auxiliar: Guardar capa de detalle en PNG
void save_detail_png(const std::string& filename, const std::vector<float>& layer, 
                     int w, int h, int channels) {
    std::vector<uint8_t> out_buffer(layer.size());
    for (size_t i = 0; i < layer.size(); ++i) {
        // Aquí sumamos 128 SOLO para proyectar el rango a 8 bits visuales
        float visual_val = layer[i] + 128.0f;
        out_buffer[i] = static_cast<uint8_t>(std::clamp(visual_val, 0.0f, 255.0f));
    }
    stbi_write_png(filename.c_str(), w, h, channels, out_buffer.data(), w * channels);
}

///////////////////////////
// PULL DE FILTROS 
//////////////////////////

// Extructura de los filtros 
//void filtro_puntual(std::vector& layer, float parametro) {
//    for (size_t i = 0; i < layer.size(); ++i) {
//        // Operación matemática sobre layer[i]
//        layer[i] = layer[i] * parametro; 
//    }
//}


//void filtro_espacial(std::vector& layer, int w, int h, int channels, float parametro) {
//    std::vector copia = layer; // buffer temporal para no sobreescribir vecinos mientras lees
//    for (int y = 0; y < h; ++y) {
//        for (int x = 0; x < w; ++x) {
//            for (int c = 0; c < channels; ++c) {
//                // Operación usando vecinos con clamp_coord
//            }
//        }
//    }
//}

// Filtro de umbralización
void filtro_escala_0(std::vector<float>& layer, float umbral) {
    for (size_t i = 0; i < layer.size(); ++i) {
        if (std::abs(layer[i]) < umbral) {
            layer[i] = 0.0f;
        }
    }
}


// Ganancia de volumen
void filtro_escala_1(std::vector<float>& layer, float ganancia) {
    for (size_t i = 0; i < layer.size(); ++i) {
        layer[i] *= ganancia;
    }
}

// Atenuacion de volumenes grandes
void filtro_escala_2(std::vector<float>& layer, float factor) {
    for (size_t i = 0; i < layer.size(); ++i) {
        layer[i] *= factor;
    }
}


// Limitador
void filtro_escala_3(std::vector<float>& layer, float limite) {
    for (size_t i = 0; i < layer.size(); ++i) {
        layer[i] = std::clamp(layer[i], -limite, limite);
    }
}


// Atenuacion de volumenes grandes
void filtro_escala_4(std::vector<float>& layer, float escala) {
    for (size_t i = 0; i < layer.size(); ++i) {
        layer[i] *= escala;
    }
}


// Filtro residual: gamma correction
void filtro_residual(std::vector<float>& residual, float gamma) {
    for (size_t i = 0; i < residual.size(); ++i) {
        float norm = std::clamp(residual[i] / 255.0f, 0.0f, 1.0f);
        residual[i] = std::pow(norm, gamma) * 255.0f;
    }
}




// =============================
//            MAIN
// =============================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada> [num_escalas (def: 5)]\n";
        return 1;
    }

    // const int num_scales = 4; // Fijamos 4 escalas + 1 residual para asignar 5 filtros específicos
    int num_scales = (argc >= 3) ? std::stoi(argv[2]) : 5;
    int w, h, channels;

    unsigned char* raw_img = stbi_load(argv[1], &w, &h, &channels, 0);
    if (!raw_img) {
        std::cerr << "Error al abrir la imagen: " << argv[1] << "\n";
        return 1;
    }

    size_t total_pixels = static_cast<size_t>(w) * h * channels;
    std::vector<float> current(total_pixels);
    for (size_t i = 0; i < total_pixels; ++i) {
        current[i] = static_cast<float>(raw_img[i]);
    }
    stbi_image_free(raw_img);

    // Creamos directorio de resultados
    fs::create_directories("capas_filtradas");
    fs::create_directories("capas");

    // =============================
    // ETAPA 1: Descomposición por frecuencia usando convolución a trous
    // ============================
    // Guardamos cada capa en memoria
    std::vector<float> blurred(total_pixels);
    std::vector<std::vector<float>> detail_layers(num_scales, std::vector<float>(total_pixels));

    for (int s = 0; s < num_scales; ++s) {
        atrous_blur(current, blurred, w, h, channels, s);

        // Capa de detalle detalle puro centrado
        for (size_t i = 0; i < total_pixels; ++i) {
            detail_layers[s][i] = current[i] - blurred[i];
        }
        
        std::string filename = "capas/escala_" + std::to_string(s + 1) + ".png";
        save_detail_png(filename, detail_layers[s], w, h, channels);
        std::cout << "Generada: " << filename << "\n";
        current = blurred;
    }
    
    // La capa residual final es la última imagen residual
    std::vector<float>& residual = current;

    // Guardamos también la residual en PNG (sin +128 porque no es detalle)
    std::vector<uint8_t> res_buffer(total_pixels);
    for (size_t i = 0; i < total_pixels; ++i) {
        res_buffer[i] = static_cast<uint8_t>(std::clamp(residual[i], 0.0f, 255.0f));
    }
    std::string filename_residual = "capas/residual.png";
    stbi_write_png(filename_residual.c_str(), w, h, channels, res_buffer.data(), w * channels);
    std::cout << "Generada: " << filename_residual << "\n";

    // =============================
    // ETAPA 2: Paralelismo funcional
    // =============================

    std::cout << "Lanzando hilos con tareas funcionales distintas...\n";

    for (int s = 0; s < num_scales; ++s) {
        std::string filename = "capas_filtradas/escala_" + std::to_string(s + 1) + ".png";

        int capa = s % 5; // 0, 1, 2, 3, 4 para asignar filtros específicos

        if (capa == 0) {
            filtro_escala_0(detail_layers[s], 4.0f);
        } else if (capa == 1) {
            filtro_escala_1(detail_layers[s], 1.5f);
        } else if (capa == 2) {
            filtro_escala_2(detail_layers[s], 1.2f);
        } else if (capa == 3) {
            filtro_escala_3(detail_layers[s], 50.0f);
        } else if (capa == 4) {
            filtro_escala_4(detail_layers[s], 0.9f);
        } else {
            std::cerr << "Error\n";
        }

        // Guardamos la capa filtrada en PNG
        std::string file_filt = "capas_filtradas/escala_" + std::to_string(s + 1) + "_filtrada.png";
        save_detail_png(file_filt, detail_layers[s], w, h, channels);
        std::cout << "Generada: " << file_filt << "\n";
    }

    filtro_residual(residual, 0.9f); 

    std::vector<uint8_t> res_buf(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        res_buf[i] = static_cast<uint8_t>(std::clamp(residual[i], 0.0f, 255.0f));
    }
    std::string filename_residual_filtrado = "capas_filtradas/residual_filtrado.png";
    stbi_write_png(filename_residual_filtrado.c_str(), w, h, channels, res_buf.data(), w * channels);
    std::cout << "Generada: " << filename_residual_filtrado << "\n";

    std::cout << "Todos los filtros han finalizado secuencialmente.\n";

    // =============================
    // ETAPA 3: Reconstruccion de la imagen original
    // =============================

    // Imagen final = residual + capa0 + capa1 + capa2 ...
    std::vector final_image = residual;
    for (int s = 0; s < num_scales; ++s) {
        for (size_t i = 0; i < total_pixels; ++i) {
            final_image[i] += detail_layers[s][i];
        }
    }


    std::vector<uint8_t> out_final(total_pixels);
    for (size_t i = 0; i < total_pixels; ++i) {
        out_final[i] = static_cast<uint8_t>(std::clamp(final_image[i], 0.0f, 255.0f));
    }
    stbi_write_png("resultado_final.png", w, h, channels, out_final.data(), w * channels);
    std::cout << "Resultado guardado en resultado_final.png\n";


    return 0;
}
