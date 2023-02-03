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
// Auxiliary Routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/unghr.hpp>
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
    n = GENERATE(5, 10, 30, 50, 100, 125, 150, 250,  300, 400, 500);
    gen.seed(seed);
    const real_t eps = uroundoff<real_t>(); 
    const T tol = T( 100 * n *  eps);

    const T one = T(1);
    const T zeroT = T(0);

    // Function
    Create<matrix_t> new_matrix; // For Real matrices
    // There may be an easier way of writing this, however it follows the
    // structure of a matrix typing.
    Create<legacyMatrix<std::complex<real_t>,std::size_t,Layout::ColMajor>> new_matrixC; // For Complex matrices


    // Create matrices
    std::vector<T> A_; auto A = new_matrix( A_, n, n);
    std::vector<T> H_; auto H = new_matrix( H_, n, n);
    std::vector<T> U_; auto U = new_matrix( U_, n, n);
    std::vector<T> Z_; auto Z = new_matrix( Z_, n, n);
    std::vector<T> Q_; auto Q = new_matrix( Q_, n, n);
    std::vector<T> wr(n);
    std::vector<T> wi(n);
    for (idx_t i = 0; i < n; i++) {
        wr[i] = rand_helper<T>(gen);
        wi[i] = rand_helper<T>(gen);
    }

    // Generate our matrix A as a full matrix and then reduce it to 
    // hessenberg form
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            T val = rand_helper<T>(gen);
            A(i,j) = val;
            H(i,j) = val;
        }
    }
    // Perform Hessenberg reduction on A.
    std::vector<T> tau(n);
    gehrd(0, n - 1, H, tau);
    lacpy(Uplo::General, H, Q);
    unghr(0, n - 1, Q, tau);
    // At this point, We assume that A = QHQ^H works as expected
    // zero out the parts of H that represent Q
    // IE the 'reflectors'
    // After running tests, this is necessary for our function to run.
    for (idx_t i = 1; i < n; i++) 
        for (idx_t j = 0; j < i - 1; j++) 
            H(i,j) = zeroT;

    // Since we are trying to test a generalized eigensolver, we start Z as Q so that at the end A = QUQ^-1
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            Z(i,j) = Q(i,j);

    //Call hqr
    real_t norm = real_t(0.0);
    idx_t retCode = hqr(H, 0, n - 1, wr, wi, true, Z, norm);
    CHECK(retCode == 0);

    // Getting here means that we have successfully ran all of 
    retCode = hqr_schurToEigen(H, 0, n - 1, wr, wi, Z, norm);
    CHECK(retCode == 0);
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
