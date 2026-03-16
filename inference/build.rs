fn main() {
    // OpenBLAS 링크
    println!("cargo:rustc-link-lib=openblas");

    let mut build = cc::Build::new();

    if cfg!(feature = "avx2-only") {
        // AVX2-only 백엔드 (VNNI 미사용)
        build.file("src/i8_kernel_avx2.c")
            .flag("-march=native")
            .flag("-mtune=native")
            .flag("-mno-avxvnni")   // 명시적으로 VNNI 비활성화
            .flag("-O3")
            .flag("-mavx2")
            .flag("-fopenmp")
            .flag("-funroll-loops")
            .flag("-ffast-math")
            .flag("-mprefer-vector-width=256");
    } else {
        // AVX-VNNI 백엔드 (기본)
        build.file("src/i8_kernel_vnni.c")
            .flag("-march=native")
            .flag("-mtune=native")
            .flag("-O3")
            .flag("-mavx2")
            .flag("-mavxvnni")
            .flag("-fopenmp")
            .flag("-funroll-loops")
            .flag("-ffast-math")
            .flag("-mprefer-vector-width=256");
    }

    build.compile("i8_kernel");

    // OpenMP 링크
    println!("cargo:rustc-link-lib=gomp");
}
