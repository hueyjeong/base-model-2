fn main() {
    // OpenBLAS 링크
    println!("cargo:rustc-link-lib=openblas");

    // AVX-VNNI int8 커널 컴파일 (최적화 플래그 강화)
    cc::Build::new()
        .file("src/i8_kernel.c")
        .flag("-march=native")
        .flag("-mtune=native")
        .flag("-O3")
        .flag("-mavx2")
        .flag("-mavxvnni")
        .flag("-fopenmp")
        .flag("-funroll-loops")
        .flag("-ffast-math")
        .flag("-mprefer-vector-width=256")
        .compile("i8_kernel");

    // OpenMP 링크
    println!("cargo:rustc-link-lib=gomp");
}
