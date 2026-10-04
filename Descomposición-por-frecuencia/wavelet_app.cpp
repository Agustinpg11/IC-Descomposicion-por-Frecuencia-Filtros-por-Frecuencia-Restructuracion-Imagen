// =======================================================================================
// Práctica 2 - Ingeniería de los Computadores
// Procesado de una secuencia de fotogramas por descomposición en frecuencias (à trous).
// Versión SECUENCIAL (sin hilos ni OpenMP), estructurada para paralelizarla después.
//
// Uso: ./filtros <imagen> [num_escalas (def: 5)] [guardar_png (0/1, def: 0)]
//                         [radio_bilateral (def: 1)] [num_fotogramas (def: 6)]
//
// La secuencia se simula con una ventana que se desplaza 8 píxeles por fotograma sobre la
// imagen de entrada (como un travelling de cámara). Cada fotograma es independiente.
//
// GRAFO DE TAREAS DE CADA FOTOGRAMA k
//
//   [E0 entrada] -> [E1 descomposición] -+-> [E2a filtros por capa (N tareas)] -+-> [E3 reconstrucción] -> [E4 salida]
//                    (cadena de blurs)   |                                      |                            (en orden)
//                                        +-> [E2b bilateral sobre el residual] -+
//
// PARALELISMO FUNCIONAL (distintas tareas a la vez)
//   - Dentro de un fotograma: E2a (los N filtros por capa son independientes entre sí) y
//     E2b (bilateral) son independientes; cada filtro puede empezar en cuanto su capa
//     está calculada, sin esperar a las siguientes escalas de E1.
//   - Entre fotogramas (pipeline): mientras E2 procesa el fotograma k, E1 procesa el k+1
//     y E3/E4 terminan el k-1. E4 debe respetar el orden de los fotogramas.
// PARALELISMO DE DATOS (misma operación sobre datos distintos)
//   - En todas las tareas, cada fila / píxel / canal de salida se calcula sin depender
//     de los demás (lecturas de un buffer y escritura en otro).
// Los parámetros num_escalas, radio_bilateral, num_fotogramas y el tamaño de la imagen
// permiten estudiar la escalabilidad con el tamaño del problema.
// =======================================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include <chrono>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

// Cronometraje de alta precisión (ms). Uso: auto t0 = clk::now(); ...; ms_desde(t0);
using clk = std::chrono::steady_clock;
static double ms_desde(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// =======================================================================================
// UTILIDADES
// =======================================================================================

// Reflejo periódico en los bordes (válido para cualquier desplazamiento, incluso mayor
// que el tamaño de la imagen: el paso de las escalas altas es 2^s).
inline int clamp_coord(int pos, int max_size) {
    if (pos >= 0 && pos < max_size) return pos;      // camino rápido
    if (max_size == 1) return 0;
    int periodo = 2 * (max_size - 1);
    pos %= periodo;
    if (pos < 0) pos += periodo;
    return (pos < max_size) ? pos : periodo - pos;
}

// Huella de 64 bits (FNV-1a). Se puede encadenar pasando el valor anterior en 'h'.
const uint64_t FNV_BASE = 1469598103934665603ULL;
uint64_t fnv1a(const std::vector<uint8_t>& datos, uint64_t h = FNV_BASE) {
    for (uint8_t b : datos) { h ^= b; h *= 1099511628211ULL; }
    return h;
}

// Guardar una capa de detalle en PNG (se suma 128 solo para visualizarla en 8 bits)
void save_detail_png(const std::string& filename, const std::vector<float>& layer,
                     int w, int h, int channels) {
    std::vector<uint8_t> out_buffer(layer.size());
    for (size_t i = 0; i < layer.size(); ++i) {
        out_buffer[i] = static_cast<uint8_t>(std::clamp(layer[i] + 128.0f, 0.0f, 255.0f));
    }
    stbi_write_png(filename.c_str(), w, h, channels, out_buffer.data(), w * channels);
}

// Guardar una capa de intensidad (sin +128) en PNG
void save_intensity_png(const std::string& filename, const std::vector<float>& layer,
                        int w, int h, int channels) {
    std::vector<uint8_t> out_buffer(layer.size());
    for (size_t i = 0; i < layer.size(); ++i) {
        out_buffer[i] = static_cast<uint8_t>(std::clamp(layer[i], 0.0f, 255.0f));
    }
    stbi_write_png(filename.c_str(), w, h, channels, out_buffer.data(), w * channels);
}

// =======================================================================================
// E1: DESCOMPOSICIÓN À TROUS
// =======================================================================================

// Blur à trous separable (B-spline cúbico: 1/16, 4/16, 6/16, 4/16, 1/16 con huecos de 2^scale).
// Paso horizontal al buffer temporal 'temp' y después paso vertical a 'dst'.
void atrous_blur(const std::vector<float>& src, std::vector<float>& dst, std::vector<float>& temp,
                 int w, int h, int channels, int scale) {
    int step = 1 << scale;

    // 1. Paso horizontal
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int x_m2 = clamp_coord(x - 2 * step, w);
            int x_m1 = clamp_coord(x - 1 * step, w);
            int x_p1 = clamp_coord(x + 1 * step, w);
            int x_p2 = clamp_coord(x + 2 * step, w);
            for (int c = 0; c < channels; ++c) {
                temp[(y * w + x) * channels + c] =
                      0.375f  * src[(y * w + x) * channels + c]
                    + 0.25f   * (src[(y * w + x_m1) * channels + c] + src[(y * w + x_p1) * channels + c])
                    + 0.0625f * (src[(y * w + x_m2) * channels + c] + src[(y * w + x_p2) * channels + c]);
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
                dst[(y * w + x) * channels + c] =
                      0.375f  * temp[(y * w + x) * channels + c]
                    + 0.25f   * (temp[(y_m1 * w + x) * channels + c] + temp[(y_p1 * w + x) * channels + c])
                    + 0.0625f * (temp[(y_m2 * w + x) * channels + c] + temp[(y_p2 * w + x) * channels + c]);
            }
        }
    }
}

// =======================================================================================
// E2a: FILTROS POR ESCALA (cada capa de detalle usa un filtro distinto; son independientes
// entre sí -> paralelismo funcional)
// =======================================================================================

// Auxiliar: energía local media (media de v^2 en ventana (2r+1)x(2r+1), separable).
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

// Escala 0 (detalle más fino): reducción de ruido tipo Wiener adaptativo.
// Ganancia (e - sigma^2)/e según la energía local e (ventana 5x5); 0 si e <= sigma^2.
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

// Escala 1: realce de bordes. Magnitud del gradiente de Sobel (3x3); ganancia 1 + k*t,
// con t = min(1, |grad| / 64).
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

// Escala 2: contraste local. Lleva la energía RMS local (ventana 7x7) hacia un objetivo (12).
// Ganancia g en [0.5, 2]; factor aplicado = 1 + fuerza*(g-1). Con fuerza > 1 se extrapola
// más allá de g (fuerza = 1.2 es un valor deliberado; con 0.5 el efecto sería más suave).
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

// Escala 3: limitador suave (tanh). Satura los valores grandes hacia +-limite.
void filtro_escala_3(std::vector<float>& layer, float limite) {
    for (size_t i = 0; i < layer.size(); ++i) {
        layer[i] = limite * std::tanh(layer[i] / limite);
    }
}

// Escala 4 (más gruesa): mediana 3x3 (elimina valores atípicos) y atenuación por 'escala'.
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

// =======================================================================================
// E2b: FILTRO BILATERAL (implementación directa, sin optimizar) - se aplica al residual.
// Cada píxel/canal se promedia con su ventana (2r+1)x(2r+1) ponderando por distancia
// espacial y por diferencia de valor (preserva bordes).
//
// Coste: O(píxeles * canales * (2r+1)^2), con 2 llamadas a exp() por vecino.
// Notas para la memoria:
//  - 2.2: los accesos a vecinos saltan filas (stride w*channels); los píxeles de salida
//    son independientes entre sí.
//  - 2.3: el tiempo crece con píxeles y con (2r+1)^2.
//  - 2.4: comprobar con -fopt-info-vec-optimized/-missed si GCC vectoriza los bucles dx/dy
//    (con GCC 13.3 en -O3 NO lo hacía: "control flow in loop" por clamp_coord).
//  - Paralelismo de datos: cada (y, x, c) se puede calcular en paralelo.
// =======================================================================================
void filtro_bilateral_naive_residual(const std::vector<float>& src, std::vector<float>& dst,
                                     int w, int h, int channels, int radio,
                                     float sigma_espacial, float sigma_color) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                float suma_pesos = 0.0f;
                float suma_valores = 0.0f;
                float valor_centro = src[(y * w + x) * channels + c];

                for (int dy = -radio; dy <= radio; ++dy) {
                    int ny = clamp_coord(y + dy, h);
                    for (int dx = -radio; dx <= radio; ++dx) {
                        int nx = clamp_coord(x + dx, w);

                        float valor_vecino = src[(ny * w + nx) * channels + c];
                        float dist_espacial = static_cast<float>(dx * dx + dy * dy);
                        float dist_color = (valor_vecino - valor_centro) * (valor_vecino - valor_centro);

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

// =======================================================================================
// FOTOGRAMA Y ETAPAS
// =======================================================================================

struct Fotograma {
    int w = 0, h = 0, channels = 0;
    std::vector<float> imagen;                  // entrada (float); se libera tras E1
    std::vector<std::vector<float>> detalle;    // capas de detalle (E1 -> E2a -> E3)
    std::vector<float> residual;                // capa residual (E1 -> E2b -> E3)
    std::vector<uint8_t> salida;                // imagen final de 8 bits (E3 -> E4)
};

struct Tiempos {                                // ms acumulados sobre todos los fotogramas
    double entrada = 0, descomposicion = 0, filtros = 0, bilateral = 0,
           reconstruccion = 0, salida = 0, png = 0;
    std::vector<double> filtro_capa;            // tiempo de cada filtro por escala
};

// E0: ventana de tamaño (fw x fh) desplazada k*desplaz píxeles sobre la imagen original
Fotograma etapa_entrada(const unsigned char* raw, int W, int channels, int fw, int fh,
                        int k, int desplaz) {
    Fotograma f;
    f.w = fw; f.h = fh; f.channels = channels;
    f.imagen.resize(static_cast<size_t>(fw) * fh * channels);
    const int ox = k * desplaz, oy = k * desplaz;
    for (int y = 0; y < fh; ++y) {
        for (int x = 0; x < fw; ++x) {
            for (int c = 0; c < channels; ++c) {
                f.imagen[(static_cast<size_t>(y) * fw + x) * channels + c] =
                    static_cast<float>(raw[(static_cast<size_t>(y + oy) * W + (x + ox)) * channels + c]);
            }
        }
    }
    return f;
}

// E1: descomposición en num_scales capas de detalle + residual.
// Cada blur usa el resultado del anterior -> cadena secuencial (dependencia entre escalas).
void etapa_descomposicion(Fotograma& f, int num_scales) {
    const size_t total = f.imagen.size();
    std::vector<float> current = std::move(f.imagen);
    std::vector<float> blurred(total), temp(total);
    f.detalle.assign(num_scales, std::vector<float>(total));

    for (int s = 0; s < num_scales; ++s) {
        atrous_blur(current, blurred, temp, f.w, f.h, f.channels, s);
        for (size_t i = 0; i < total; ++i) {
            f.detalle[s][i] = current[i] - blurred[i];
        }
        std::swap(current, blurred);            // 'current' pasa a ser la imagen difuminada
    }
    f.residual = std::move(current);            // la residual es el último blur
    f.imagen.clear();
    f.imagen.shrink_to_fit();
}

// E2a: un filtro distinto por escala (se repiten cada 5 escalas). Las capas son
// independientes: cada filtro solo lee y escribe su propia capa.
void etapa_filtros_capas(Fotograma& f, Tiempos& t) {
    const int num_scales = static_cast<int>(f.detalle.size());
    for (int s = 0; s < num_scales; ++s) {
        auto t0 = clk::now();
        switch (s % 5) {
            case 0:  filtro_escala_0(f.detalle[s], f.w, f.h, f.channels, 4.0f); break;
            case 1:  filtro_escala_1(f.detalle[s], f.w, f.h, f.channels, 1.5f); break;
            case 2:  filtro_escala_2(f.detalle[s], f.w, f.h, f.channels, 1.2f); break;
            case 3:  filtro_escala_3(f.detalle[s], 50.0f); break;
            default: filtro_escala_4(f.detalle[s], f.w, f.h, f.channels, 0.9f); break;
        }
        t.filtro_capa[s] += ms_desde(t0);
    }
}

// E2b: bilateral sobre el residual (sigma espacial 3, sigma de color 25)
void etapa_bilateral(Fotograma& f, int radio) {
    std::vector<float> filtrado(f.residual.size());
    filtro_bilateral_naive_residual(f.residual, filtrado, f.w, f.h, f.channels, radio, 3.0f, 25.0f);
    f.residual = std::move(filtrado);
}

// E3: imagen final = residual filtrado + suma de capas de detalle filtradas (8 bits)
void etapa_reconstruccion(Fotograma& f) {
    for (const auto& capa : f.detalle) {
        for (size_t i = 0; i < f.residual.size(); ++i) {
            f.residual[i] += capa[i];
        }
    }
    f.salida.resize(f.residual.size());
    for (size_t i = 0; i < f.residual.size(); ++i) {
        f.salida[i] = static_cast<uint8_t>(std::clamp(f.residual[i], 0.0f, 255.0f));
    }
}

// =============================
//            MAIN
// =============================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada> [num_escalas (def: 5)] [guardar_png (0/1, def: 0)]"
                     " [radio_bilateral (def: 1)] [num_fotogramas (def: 6)]\n";
        return 1;
    }

    const int  num_scales      = (argc >= 3) ? std::stoi(argv[2]) : 5;
    const bool guardar_png     = (argc >= 4) ? (std::stoi(argv[3]) != 0) : false;
    const int  radio_bilateral = (argc >= 5) ? std::stoi(argv[4]) : 1;
    const int  num_frames      = (argc >= 6) ? std::stoi(argv[5]) : 6;
    const int  desplaz         = 8;   // píxeles que se desplaza la ventana entre fotogramas
    if (num_scales < 1 || radio_bilateral < 0 || num_frames < 1) {
        std::cerr << "Error: num_escalas >= 1, radio_bilateral >= 0 y num_fotogramas >= 1\n";
        return 1;
    }

    int W, H, channels;
    unsigned char* raw_img = stbi_load(argv[1], &W, &H, &channels, 0);
    if (!raw_img) {
        std::cerr << "Error al abrir la imagen: " << argv[1] << "\n";
        return 1;
    }
    const int fw = W - (num_frames - 1) * desplaz;
    const int fh = H - (num_frames - 1) * desplaz;
    if (fw < 16 || fh < 16) {
        std::cerr << "Error: imagen demasiado pequeña para " << num_frames << " fotogramas\n";
        stbi_image_free(raw_img);
        return 1;
    }

    if (guardar_png) {
        fs::create_directories("capas_filtradas");
        fs::create_directories("capas");
    }

    Tiempos t;
    t.filtro_capa.assign(num_scales, 0.0);
    uint64_t hash_global = FNV_BASE;
    auto t_inicio_total = clk::now();

    std::cout << "Procesando " << num_frames << " fotogramas de " << fw << "x" << fh
              << " (" << channels << " canales), " << num_scales << " escalas, radio bilateral "
              << radio_bilateral << "\n";

    for (int k = 0; k < num_frames; ++k) {
        const bool png_este = guardar_png && (k == 0);   // solo se guardan capas del fotograma 0
        auto t0 = clk::now();

        // E0: entrada
        Fotograma f = etapa_entrada(raw_img, W, channels, fw, fh, k, desplaz);
        t.entrada += ms_desde(t0);

        // E1: descomposición
        t0 = clk::now();
        etapa_descomposicion(f, num_scales);
        t.descomposicion += ms_desde(t0);

        if (png_este) {
            t0 = clk::now();
            for (int s = 0; s < num_scales; ++s)
                save_detail_png("capas/escala_" + std::to_string(s + 1) + ".png", f.detalle[s], f.w, f.h, f.channels);
            save_intensity_png("capas/residual.png", f.residual, f.w, f.h, f.channels);
            t.png += ms_desde(t0);
        }

        // E2a y E2b: tareas independientes entre sí
        t0 = clk::now();
        etapa_filtros_capas(f, t);
        t.filtros += ms_desde(t0);

        t0 = clk::now();
        etapa_bilateral(f, radio_bilateral);
        t.bilateral += ms_desde(t0);

        if (png_este) {
            t0 = clk::now();
            for (int s = 0; s < num_scales; ++s)
                save_detail_png("capas_filtradas/escala_" + std::to_string(s + 1) + "_filtrada.png", f.detalle[s], f.w, f.h, f.channels);
            save_intensity_png("capas_filtradas/residual_filtrado.png", f.residual, f.w, f.h, f.channels);
            t.png += ms_desde(t0);
        }

        // E3: reconstrucción
        t0 = clk::now();
        etapa_reconstruccion(f);
        t.reconstruccion += ms_desde(t0);

        // E4: salida en orden (checksum encadenado; PNG solo del fotograma 0)
        t0 = clk::now();
        uint64_t hash_frame = fnv1a(f.salida);
        hash_global = fnv1a(f.salida, hash_global);
        t.salida += ms_desde(t0);
        std::cout << "Fotograma " << k << ": 0x" << std::hex << hash_frame << std::dec << "\n";

        if (png_este) {
            t0 = clk::now();
            stbi_write_png("resultado_final.png", f.w, f.h, f.channels, f.salida.data(), f.w * f.channels);
            t.png += ms_desde(t0);
        }
    }
    stbi_image_free(raw_img);

    const double t_total = ms_desde(t_inicio_total);
    std::cout << "Checksum FNV-1a global: 0x" << std::hex << hash_global << std::dec << "\n\n";

    // ----- Resumen -----
    double suma_filtros = 0.0;
    for (double v : t.filtro_capa) suma_filtros += v;
    const double e1 = t.entrada + t.descomposicion;   // etapa de entrada y descomposición
    const double e3 = t.reconstruccion + t.salida;    // reconstrucción y salida
    const double trabajo = e1 + t.filtros + t.bilateral + e3;   // tiempo de cálculo sin PNG

    std::cout << "========================================\n";
    std::cout << "          RESUMEN DE RENDIMIENTO        \n";
    std::cout << "========================================\n";
    std::cout << "Modo guardado PNG : " << (guardar_png ? "ACTIVADO" : "DESACTIVADO (Benchmark)") << "\n";
    std::cout << "Fotogramas        : " << num_frames << " de " << fw << "x" << fh << "\n";
    std::cout << "Escalas procesadas: " << num_scales << "\n";
    std::cout << "Radio bilateral   : " << radio_bilateral << " (ventana " << 2 * radio_bilateral + 1
              << "x" << 2 * radio_bilateral + 1 << ")\n";
    std::cout << "----------------------------------------\n";
    std::cout << "E0 Entrada               : " << t.entrada << " ms\n";
    std::cout << "E1 Descomposición        : " << t.descomposicion << " ms\n";
    std::cout << "E2a Filtros por capa     : " << t.filtros << " ms\n";
    for (int s = 0; s < num_scales; ++s)
        std::cout << "      filtro_escala_" << s % 5 << " (escala " << s << "): " << t.filtro_capa[s] << " ms\n";
    std::cout << "E2b Bilateral (residual) : " << t.bilateral << " ms\n";
    std::cout << "E3 Reconstrucción        : " << t.reconstruccion << " ms\n";
    std::cout << "E4 Salida (checksum)     : " << t.salida << " ms\n";
    if (guardar_png) std::cout << "PNG (fuera del análisis) : " << t.png << " ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Tiempo Total Algoritmo   : " << t_total << " ms (" << t_total / 1000.0 << " s)\n";
    std::cout << "----------------------------------------\n";

    // Cotas teóricas de speedup (ley de Amdahl) a partir de los tiempos medidos
    const double mayor_etapa = std::max({e1, t.filtros, t.bilateral, e3});
    const double camino_frame = e1 + std::max(t.filtros, t.bilateral) + e3;
    std::cout << "Peso bilateral / (filtros+bilateral): " << 100.0 * t.bilateral / (t.filtros + t.bilateral) << " %\n";
    std::cout << "Speedup funcional máx. dentro de un fotograma (E2a || E2b)    : " << trabajo / camino_frame << "x\n";
    std::cout << "Speedup funcional máx. con pipeline de 4 etapas entre fotogramas: " << trabajo / mayor_etapa << "x\n";
    std::cout << "========================================\n";

    return 0;
}
