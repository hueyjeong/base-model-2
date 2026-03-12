fn main() {
    // OpenBLAS 링크
    println!("cargo:rustc-link-lib=openblas");

    // AVX-VNNI int8 커널 컴파일
    cc::Build::new()
        .file("src/i8_kernel.c")
        .flag("-march=native")
        .flag("-O3")
        .flag("-mavx2")
        .flag("-mavxvnni")
        .flag("-fopenmp")
        .compile("i8_kernel");

    // OpenMP 링크
    println!("cargo:rustc-link-lib=gomp");
}
