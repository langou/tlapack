/// @file test_hqr_schurToEigen.cpp
/// @brief Test HQR. 
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "testutils.hpp"
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/getri.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/blas/gemm.hpp>

#include <tlapack/lapack/hqr.hpp>
#include <tlapack/lapack/hqr_schurToEigen.hpp>

// Auxiliary routines

using namespace tlapack;

TEMPLATE_TEST_CASE("schur form is backwards stable", "[hqr][schur]", TLAPACK_REAL_TYPES_TO_TEST)
{
    srand(1);
    rand_generator gen;
    

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;
    // Complex numbers are constructed as complex_t(real,imaginary)
    typedef complex_type<T> complex_t; 

    idx_t n;
    //auto matrix_type = GENERATE(as<std::string>{}, "Full Matrix", "Inner Window");
    idx_t seed = GENERATE(123,623,134,5); // Numbers generated by my shell's random
    //n = GENERATE(5, 10, 30);//, 50, 100, 125, 150, 250,  300, 400, 500);
    n = 100;
    gen.seed(seed);
    const real_t eps = uroundoff<real_t>(); 
    const T tol = T( 10 * sqrt(n) *  eps);

    const T one = T(1);
    const T zeroT = T(0);

    // Function
    Create<matrix_t> new_matrix; // For Real matrices
    // There may be an easier way of writing this, however it follows the
    // structure of a matrix typing.
    Create<legacyMatrix<std::complex<real_t>,std::size_t,Layout::ColMajor>> new_matrixC; // For Complex matrices


    // Create matrices
    std::vector<T> A_; auto A = new_matrix( A_, n, n);
    std::vector<T> U_; auto U = new_matrix( U_, n, n);
    std::vector<T> Z_; auto Z = new_matrix( Z_, n, n);
    std::vector<T> wr(n);
    std::vector<T> wi(n);
    for (idx_t i = 0; i < n; i++) {
        wr[i] = rand_helper<T>(gen);
        wi[i] = rand_helper<T>(gen);
    }

    // Generate our matrix A as a full matrix and then reduce it to 
    // hessenberg form
    for (idx_t i = 0; i < n; i++)
       for (idx_t j = 0; j < n; j++)
          A(i,j) = rand_helper<T>(gen); 

    // Perform Hessenberg reduction on A.
    std::vector<T> tau(n);
    gehrd(0, n - 1, A, tau);
    lacpy(Uplo::Lower, A, Q);
    unghr(0, n - 1, Q, tau);
    // zero out the parts of A that represent Q
    // IE the 'reflectors'
    // After running tests, this is necessary for our function to run.
    for (idx_t i = 0; i < n; i++) 
        if (i != 0)
            for (idx_t j = low; j < i - 1; j++) 
                A(i,j) = zeroT;
    // If we want to test the ilo and igh behavior we 
    // Copy A into U
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            U(i,j) = A(i,j);

    // Start Z as eye(n)
    for (idx_t i = 0; i < n; i++)
        Z(i,i) = one;

    //Call hqr
    real_t norm = real_t(0.0);
    idx_t retCode = hqr(U, low, igh, wr, wi, true, Z, norm);
    CHECK(retCode == 0);

    // Getting here means that we have successfully ran all of 
    retCode = hqr_schurToEigen(U, low, igh, wr, wi, Z, norm);
    CHECK(retCode == 0);
    // Zero out below quasi diagonal elements of T
    // First, zero out everything below the 1st subdiagonal
    for (idx_t i = 0; i < n; i++) 
        if (i != 0)
            for (idx_t j = 0; j < i - 1; j++) 
                U(i,j) = zeroT;
    // if wi[k]  = 0 then the sub diagonal elements need to be 0
    // If wi[k] != 0 then we have a schur block  
    idx_t k;
    for (k = 0; k < n-1; k++) {
        if (wi[k] == real_t(0)) {
            U(k+1,k) = zeroT;
        } else if (k < n-2){
            // This means we are in a schur block, so the next sub diagonal
            // element must be 0
            U(k+2,k+1) = zeroT;
            k++;
        }
    }
    // Now, currently we are only testing matrices that are supposed to be diagonalizable
    // We will test representativity by constructing a matrix Zc such that Zc contains 
    // the eigenvectors stored in Z but as complex numbers
    // Similarly, do so for Dc containing the eigenvalues stored as complex numbers
    std::vector<std::complex<real_t>> Zc_; auto Zc = new_matrixC( Zc_, n, n);
    for (idx_t j = 0; j < n; j++) { // For each column
        for (idx_t i = 0; i < n; i++) { // Grab the ith row
            // For more information on how the eigenvectors are constructed
            // see the documentation for hqr_schurToEigen
            // If wi[j] is 0, then we have a real eigenvector, so we only need to copy the current column
            if (wi[j] == zeroT)
                Zc(i,j) = std::complex<real_t>(Z(i,j), 0);
            // If wi[j] is positive, then we have an eigenvector of the form Z[:,j] + Z[:,j+1]*i
            else if (wi[j] > zeroT)
                Zc(i,j) = std::complex<real_t>(Z(i,j), Z(i,j + 1));
            // Otherwise, we found the conjugate pair so we need Z[:,j-1] - Z[:,j]*i
            else
                Zc(i,j) = std::complex<real_t>(Z(i,j - 1), -Z(i,j));
        }
    }
    std::vector<std::complex<real_t>> Zi_; auto Zi = new_matrixC( Zi_, n, n);
    lacpy(Uplo::General, Zc, Zi);
    std::vector<T> Piv(n);
    // Perform LU Decomp of Zi
    retCode = getrf(Zi,Piv);
    CHECK(retCode == 0); // Ensure we properly computed the LU
    // Now compute the inverse of Zi
    retCode = getri(Zi,Piv);
    CHECK(retCode == 0); // Ensure we properly computed the Inverse
    // Now test VDV^{-1} - A
    std::vector<std::complex<real_t>> lhs_; auto lhs = new_matrixC( lhs_, n, n);
    std::vector<std::complex<real_t>> Dc_; auto Dc = new_matrixC( Dc_, n, n);
    for (idx_t i = 0; i < n; i++)
        Dc(i,i) = std::complex<real_t>(wr[i], wi[i]);
    // We need to also construct Ac which is just a complex equivalent of A
    // with 0 for all imaginary parts
    std::vector<std::complex<real_t>> Ac_; auto Ac = new_matrixC( Ac_, n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            Ac(i,j) = std::complex<real_t>(A(i,j),0);

    gemm(Op::NoTrans, Op::NoTrans, real_t(1), Zc, Dc, lhs);
    // Note: This overwrites A, but we already have the norm of A saved from hqr
    gemm(Op::NoTrans, Op::NoTrans, real_t(1), lhs, Zi, real_t(-1), Ac);
    // Compute the frobenius norm of the residual
    real_t normR = lange(tlapack::frob_norm, Ac);
    real_t normZ = lange(tlapack::frob_norm, Zc);
    real_t normZi = lange(tlapack::frob_norm, Zi);
    real_t normD = lange(tlapack::frob_norm, Dc);
    CHECK(normR <= tol * normZ * normZi * normD);
    //CHECK(normR <= tol * norm);
}
