#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <filesystem>
namespace fs = std::filesystem;


// =======================================================================================
// Función auxiliar para medir el tiempo de ejecución con alta precisión (en milisegundos).
// 
// ¿Cómo y para qué la vamos a utilizar en esta práctica?
// - Cronometraje por fases: Nos permite aislar el tiempo de la Etapa 1 (blur), la 
//   Etapa 2 (filtros espaciales) y la Etapa 3 (reconstrucción).
// - Aislar el cuello de botella (I/O): Es crucial para separar el tiempo real de cálculo 
//   de la CPU del tiempo que tarda el disco duro en guardar las imágenes PNG.
// - Toma de métricas: Nos dará los datos exactos para rellenar las tablas de la Memoria 
//   al comparar cómo escalan los tiempos según el tamaño del problema y las banderas de 
//   compilación de GCC (-O0, -O1, -O3, autovectorización).
//
// Uso típico: 
//   auto t0 = clk::now(); 
//   /* ... código a medir ... */ 
//   double milisegundos = ms_desde(t0);
// =======================================================================================

using clk = std::chrono::steady_clock;
static double ms_desde(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}
 

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

///// FILTROS QUE YO AÑADIRÍA PARA ELEVAR COMPUTACIONALMENTE EL PROBLEMA

// FUNCIÓN AUXILIAR QUE VAMOS A UTILIZAR PARA LOS FILTROS : 

/*
void energia_local(const std::vector<float>& layer, std::vector<float>& out,
                   int w, int h, int channels, int r) {
    std::vector<float> tmp(layer.size());
    const float inv = 1.0f / static_cast<float>(2 * r + 1);
 
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -r; k <= r; ++k) {
                    float v = layer[(y * w + clamp_coord(x + k, w)) * channels + c];
                    acc += v * v;
                }
                tmp[(y * w + x) * channels + c] = acc * inv;
            }
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -r; k <= r; ++k) {
                    acc += tmp[(clamp_coord(y + k, h) * w + x) * channels + c];
                }
                out[(y * w + x) * channels + c] = acc * inv;
            }
        }
    }
}

FILTROS : 

void filtro_escala_0(std::vector<float>& layer, int w, int h, int channels, float sigma) {
    std::vector<float> energia(layer.size());
    energia_local(layer, energia, w, h, channels, 2);
    const float s2 = sigma * sigma;
    for (size_t i = 0; i < layer.size(); ++i) {
        float e = energia[i];
        float g = (e > s2) ? (e - s2) / e : 0.0f;
        layer[i] *= g;
    }
}

void filtro_escala_1(std::vector<float>& layer, int w, int h, int channels, float k) {
    const std::vector<float> src = layer; // copia: no pisar vecinos mientras se leen
    const float gradiente_ref = 64.0f;
 
    for (int y = 0; y < h; ++y) {
        int ym = clamp_coord(y - 1, h), yp = clamp_coord(y + 1, h);
        for (int x = 0; x < w; ++x) {
            int xm = clamp_coord(x - 1, w), xp = clamp_coord(x + 1, w);
            for (int c = 0; c < channels; ++c) {
                auto S = [&](int yy, int xx) { return src[(yy * w + xx) * channels + c]; };
                float gx = (S(ym, xp) + 2.0f * S(y, xp) + S(yp, xp))
                         - (S(ym, xm) + 2.0f * S(y, xm) + S(yp, xm));
                float gy = (S(yp, xm) + 2.0f * S(yp, x) + S(yp, xp))
                         - (S(ym, xm) + 2.0f * S(ym, x) + S(ym, xp));
                float mag = std::sqrt(gx * gx + gy * gy);
                float t = std::min(1.0f, mag / gradiente_ref);
                layer[(y * w + x) * channels + c] = S(y, x) * (1.0f + k * t);
            }
        }
    }
}

void filtro_escala_2(std::vector<float>& layer, int w, int h, int channels, float fuerza) {
    std::vector<float> energia(layer.size());
    energia_local(layer, energia, w, h, channels, 3);
    const float objetivo = 12.0f;
    for (size_t i = 0; i < layer.size(); ++i) {
        float rms = std::sqrt(energia[i]);
        float g = std::clamp(objetivo / (rms + 1.0f), 0.5f, 2.0f);
        layer[i] *= 1.0f + fuerza * (g - 1.0f);
    }
}

void filtro_escala_3(std::vector<float>& layer, float limite) {
    for (size_t i = 0; i < layer.size(); ++i) {
        layer[i] = limite * std::tanh(layer[i] / limite);
    }
}

void filtro_escala_4(std::vector<float>& layer, int w, int h, int channels, float escala) {
    const std::vector<float> src = layer;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                float v[9];
                int n = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    int yy = clamp_coord(y + dy, h);
                    for (int dx = -1; dx <= 1; ++dx) {
                        v[n++] = src[(yy * w + clamp_coord(x + dx, w)) * channels + c];
                    }
                }
                std::nth_element(v, v + 4, v + 9);
                layer[(y * w + x) * channels + c] = v[4] * escala;
            }
        }
    }
}

void filtro_residual(std::vector<float>& residual, float gamma) {
    for (size_t i = 0; i < residual.size(); ++i) {
        float norm = std::clamp(residual[i] / 255.0f, 0.0f, 1.0f);
        residual[i] = std::pow(norm, gamma) * 255.0f;
    }
}

// Huella de 64 bits (FNV-1a) de la imagen final, para comprobar que los
// resultados no cambian al modificar el codigo o las opciones de compilacion.
uint64_t fnv1a(const std::vector<uint8_t>& datos) {
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t b : datos) { h ^= b; h *= 1099511628211ULL; }
    return h;
}

*/



// =============================
//            MAIN
// =============================


// IDEAAAAAA

/* 
 * TODO (Idea para comentar):
 * He estado pensando que para sacar las métricas de tiempo reales para la memoria 
 * (y ver las diferencias cuando metamos -O3 o autovectorización), el disco duro 
 * nos va a hacer un cuello de botella gigante al guardar tantos PNGs.
 * 
 * ¿Qué te parece si le añadimos al main un tercer parámetro tipo "guardar_png (1/0)"? 
 * La idea sería que si le pasamos un 0 al ejecutar, envuelva todos los 'stbi_write_png' 
 * y 'save_detail_png' de las 3 etapas en un 'if (guardar)' y se los salte. 
 * Así el programa hace todos los cálculos matemáticos pesados en la RAM, pero no 
 * escribe nada en disco y sacamos tiempos puros de CPU.
 * 
 * Échale un ojo y me dices si te mola la idea antes de ponerme a cambiar los ifs.
 */


 // otra idea

 /*
 * =========================================================================================
 * NUEVO FILTRO: BILATERAL NAIVE (EL "MONSTRUO" COMPUTACIONAL)
 * =========================================================================================
 * Ey, he metido esta función nueva aquí para aplicarla SÓLO a la capa residual en la Etapa 2.
 * 
 * ¿Por qué necesitamos esta bestia?
 * Los filtros puntuales y espaciales pequeños se ejecutan tan rápido que no íbamos a 
 * llegar al "orden de segundos" que exige el guion. Además, necesitamos código que haga 
 * sufrir a la CPU para poder medir mejoras reales luego.
 * 
 * ¿Qué hace y por qué es perfecto para la Memoria?
 * Es un Filtro Bilateral programado a "fuerza bruta" (naive). Tiene 4 bucles 'for' anidados 
 * y calcula dos exponenciales (std::exp) por cada píxel vecino. Esto nos regala medio 
 * trabajo del PDF:
 * 
 * 1. Tarea 2.2 (Memoria y Caché): Al leer los píxeles vecinos saltando por la matriz, 
 *    rompemos la localidad espacial de la caché. Es el ejemplo perfecto para comentarlo.
 * 2. Tarea 2.3 (Carga de trabajo): Si el profe nos pide gráficas, solo con subir el 
 *    'radio' de 5 a 10, o meter una foto en 4K, el tiempo se dispara de forma brutal y 
 *    la gráfica de escalabilidad sale sola.
 * 3. Tarea 2.4 (Autovectorización): Como los bucles internos son matemáticas puras sin 
 *    'ifs' raros, cuando compilemos con GCC en -O3 el compilador va a vectorizar todo 
 *    esto con instrucciones SIMD. La tabla de tiempos va a quedar espectacular.
 * 4. Paralelismo de Datos: El cálculo de cada píxel es totalmente independiente del 
 *    resto, ideal para proponer en la memoria cómo lo pasaríamos a una GPU o hilos.
 * =========================================================================================
 */

 /*
 // Filtro Bilateral Naive (Secuencial y sin optimizar para saturar la CPU)
void filtro_bilateral_naive(const std::vector<float>& src, std::vector<float>& dst, 
                            int w, int h, int channels, int radio, 
                            float sigma_espacial, float sigma_color) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                float suma_pesos = 0.0f;
                float suma_valores = 0.0f;
                float valor_centro = src[(y * w + x) * channels + c];

                // Ventana local pesada (2*radio + 1) x (2*radio + 1)
                for (int dy = -radio; dy <= radio; ++dy) {
                    for (int dx = -radio; dx <= radio; ++dx) {
                        int ny = std::max(0, std::min(h - 1, y + dy));
                        int nx = std::max(0, std::min(w - 1, x + dx));
                        
                        float valor_vecino = src[(ny * w + nx) * channels + c];
                        float dist_espacial = (dx * dx + dy * dy);
                        float dist_color = (valor_vecino - valor_centro) * (valor_vecino - valor_centro);

                        // Exponenciales (cuello de botella matemático perfecto)
                        float peso_espacial = std::exp(-dist_espacial / (2.0f * sigma_espacial * sigma_espacial));
                        float peso_color = std::exp(-dist_color / (2.0f * sigma_color * sigma_color));
                        float peso_total = peso_espacial * peso_color;

                        suma_valores += valor_vecino * peso_total;
                        suma_pesos += peso_total;
                    }
                }
                dst[(y * w + x) * channels + c] = suma_valores / suma_pesos;
            }
        }
    }
}

*/

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada> [num_escalas (def: 5)] [guardar_png (0/1, def: 0)] \n";
        return 1;
    }

    // const int num_scales = 4; // Fijamos 4 escalas + 1 residual para asignar 5 filtros específicos
    int num_scales = (argc >= 3) ? std::stoi(argv[2]) : 5;
    bool guardar_png = (argc >= 4) ? (std::stoi(argv[3]) != 0) : false;
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

    // Creamos directorio de resultados solo si vamos a guardar PNGs
    if (guardar_png) {
        fs::create_directories("capas_filtradas");
        fs::create_directories("capas");
    }

    // Cronometro global
    auto t_inicio_total = clk::now();

    // =============================
    // ETAPA 1: Descomposición por frecuencia usando convolución a trous
    // ============================

    std::cout << "--- ETAPA 1: Descomponiendo por frecuencia... ---\n";
    auto t_inicio_etapa1 = clk::now();

    // Guardamos cada capa en memoria
    std::vector<float> blurred(total_pixels);
    std::vector<std::vector<float>> detail_layers(num_scales, std::vector<float>(total_pixels));

    for (int s = 0; s < num_scales; ++s) {
        atrous_blur(current, blurred, w, h, channels, s);

        // Capa de detalle detalle puro centrado
        for (size_t i = 0; i < total_pixels; ++i) {
            detail_layers[s][i] = current[i] - blurred[i];
        }
        
        if (guardar_png) {
            std::string filename = "capas/escala_" + std::to_string(s + 1) + ".png";
            save_detail_png(filename, detail_layers[s], w, h, channels);
            std::cout << "Generada: " << filename << "\n";
        }
        current = blurred;
    }
    
    // La capa residual final es la última imagen residual
    std::vector<float>& residual = current;

    if (guardar_png) {
        // Guardamos también la residual en PNG (sin +128 porque no es detalle)
        std::vector<uint8_t> res_buffer(total_pixels);
        for (size_t i = 0; i < total_pixels; ++i) {
            res_buffer[i] = static_cast<uint8_t>(std::clamp(residual[i], 0.0f, 255.0f));
        }
        std::string filename_residual = "capas/residual.png";
        stbi_write_png(filename_residual.c_str(), w, h, channels, res_buffer.data(), w * channels);
        std::cout << "Generada: " << filename_residual << "\n";
    }

    double t_etapa1 = ms_desde(t_inicio_etapa1);
    std::cout << "Etapa 1 finalizada en: " << t_etapa1 << " ms\n\n";

    
    // =============================
    // ETAPA 2: Paralelismo funcional
    // =============================

    std::cout << "--- ETAPA 2: Aplicando filtros a las capas... ---\n";
    auto t_inicio_etapa2 = clk::now();

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

        if (guardar_png) {
        // Guardamos la capa filtrada en PNG
        std::string file_filt = "capas_filtradas/escala_" + std::to_string(s + 1) + "_filtrada.png";
        
            save_detail_png(file_filt, detail_layers[s], w, h, channels);
            std::cout << "Generada: " << file_filt << "\n";
        }
    }

    filtro_residual(residual, 0.9f); 

    if (guardar_png) {
        std::vector<uint8_t> res_buf(residual.size());
        for (size_t i = 0; i < residual.size(); ++i) {
            res_buf[i] = static_cast<uint8_t>(std::clamp(residual[i], 0.0f, 255.0f));
        }
        std::string filename_residual_filtrado = "capas_filtradas/residual_filtrado.png";
        stbi_write_png(filename_residual_filtrado.c_str(), w, h, channels, res_buf.data(), w * channels);
        std::cout << "Generada: " << filename_residual_filtrado << "\n";
    }

    double t_etapa2 = ms_desde(t_inicio_etapa2);
    std::cout << "Etapa 2 finalizada en: " << t_etapa2 << " ms\n\n";

    // =============================
    // ETAPA 3: Reconstruccion de la imagen original
    // =============================

    std::cout << "--- ETAPA 3: Recomponiendo la imagen original... ---\n";
    auto t_inicio_etapa3 = clk::now();

    // Imagen final = residual + capa0 + capa1 + capa2 ...
    std::vector final_image = residual;
    for (int s = 0; s < num_scales; ++s) {
        for (size_t i = 0; i < total_pixels; ++i) {
            final_image[i] += detail_layers[s][i];
        }
    }

    std::cout << "Imagen recompuesta...\n";

    std::vector<uint8_t> out_final(total_pixels);
    for (size_t i = 0; i < total_pixels; ++i) {
        out_final[i] = static_cast<uint8_t>(std::clamp(final_image[i], 0.0f, 255.0f));
    }
    stbi_write_png("resultado_final.png", w, h, channels, out_final.data(), w * channels);
    std::cout << "Resultado guardado en resultado_final.png\n";


    double t_etapa3 = ms_desde(t_inicio_etapa3);
    std::cout << "Etapa 3 finalizada en: " << t_etapa3 << " ms\n\n";

    double t_total = ms_desde(t_inicio_total);

    // ============================================================
    // RESUMEN DE TIEMPOS
    // ============================================================
    std::cout << "========================================\n";
    std::cout << "          RESUMEN DE RENDIMIENTO        \n";
    std::cout << "========================================\n";
    std::cout << "Modo guardado PNG : " << (guardar_png ? "ACTIVADO" : "DESACTIVADO (Benchmark)") << "\n";
    std::cout << "Escalas procesadas: " << num_scales << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Etapa 1 (Descomposición) : " << t_etapa1 << " ms\n";
    std::cout << "Etapa 2 (Filtrado)       : " << t_etapa2 << " ms\n";
    std::cout << "Etapa 3 (Reconstrucción) : " << t_etapa3 << " ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Tiempo Total Algoritmo   : " << t_total << " ms (" << t_total / 1000.0 << " s)\n";
    std::cout << "========================================\n";

    return 0;
}
