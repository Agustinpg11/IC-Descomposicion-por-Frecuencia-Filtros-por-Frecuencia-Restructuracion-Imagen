#!/bin/bash
# Compara opciones de compilación de GCC sobre el caso de referencia.
# Uso: ./bench_flags.sh            (1 ejecución por configuración)
#      REPS=3 ./bench_flags.sh     (3 ejecuciones por configuración)
SRC=wavelet_app.cpp
IMG=${IMG:-ref/imagen_ref.jpg}
ESC=${ESC:-5}
RAD=${RAD:-1}
FRM=${FRM:-6}
REPS=${REPS:-1}
OUT=bench/resultados.txt

mkdir -p bench
g++ --version | head -1 | tee $OUT
echo "Imagen: $IMG  escalas=$ESC  radio=$RAD  fotogramas=$FRM  repeticiones=$REPS" | tee -a $OUT
printf "%-20s %3s %8s %8s %8s %8s %9s  %s\n" "flags" "run" "E1_ms" "E2a_ms" "E2b_ms" "E3_ms" "total_ms" "checksum" | tee -a $OUT

configs=("-O0" "-O1" "-O2" "-O3" "-O3 -march=native" "-O3 -ffast-math")
for cfg in "${configs[@]}"; do
    name=$(echo "$cfg" | tr -d '=' | tr ' ' '_' | tr '-' 'x')
    g++ -std=c++17 $cfg $SRC -o bench/f_$name || continue
    for r in $(seq $REPS); do
        f=bench/out_${name}_$r.txt
        ./bench/f_$name $IMG $ESC 0 $RAD $FRM > $f
        e1=$(grep "^E1 " $f | awk '{print $4}')
        e2a=$(grep "^E2a " $f | awk '{print $6}')
        e2b=$(grep "^E2b " $f | awk '{print $5}')
        e3=$(grep "^E3 " $f | awk '{print $4}')
        to=$(grep "Tiempo Total" $f | awk '{print $5}')
        ck=$(grep "Checksum" $f | awk '{print $4}')
        printf "%-20s %3s %8.0f %8.0f %8.0f %8.0f %9.0f  %s\n" "$cfg" "$r" "$e1" "$e2a" "$e2b" "$e3" "$to" "$ck" | tee -a $OUT
    done
done

# Evidencia de autovectorización (solo líneas de nuestro fichero, no de stb)
g++ -std=c++17 -O3 -fopt-info-vec-optimized $SRC -o /dev/null 2>&1 | grep "^$SRC" > bench/vec_optimized.txt
g++ -std=c++17 -O3 -fopt-info-vec-missed    $SRC -o /dev/null 2>&1 | grep "^$SRC" > bench/vec_missed.txt
echo "Bucles vectorizados (-O3): $(wc -l < bench/vec_optimized.txt)  -> bench/vec_optimized.txt" | tee -a $OUT
echo "Bucles NO vectorizados   : $(wc -l < bench/vec_missed.txt)  -> bench/vec_missed.txt" | tee -a $OUT
