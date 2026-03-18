fn main() {
    let mut build = cc::Build::new();

    if cfg!(feature = "avx2-only") {
        // AVX2-only 백엔드 (VNNI 미사용)
        build.file("c_kernels/i8_kernel_avx2.c")
            .file("c_kernels/mixing_kernels.c")
            .file("c_kernels/mamba2_ssd.c")
            .flag("-march=native")
            .flag("-mtune=native")
            .flag("-mno-avxvnni")
            .flag("-O3")
            .flag("-mavx2")
            .flag("-mfma")
            .flag("-fopenmp")
            .flag("-funroll-loops")
            .flag("-mprefer-vector-width=256");
    } else {
        // AVX-VNNI 백엔드 (기본)
        build.file("c_kernels/i8_kernel_vnni.c")
            .file("c_kernels/mixing_kernels.c")
            .file("c_kernels/mamba2_ssd.c")
            .flag("-march=native")
            .flag("-mtune=native")
            .flag("-O3")
            .flag("-mavx2")
            .flag("-mfma")
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
