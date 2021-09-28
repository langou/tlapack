/// @file potrs.hpp Solves a system of linear equations A * X = B.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/potrs.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __POTRS_HH__
#define __POTRS_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"
#include "tblas.hpp"

namespace lapack {

/** Solves a system of linear equations $A X = B$ with a Hermitian
 * positive definite matrix A using the Cholesky factorization
 * $A = U^H U$ or $A = L L^H$ computed by `lapack::potrs`.
 *
 * @param[in] uplo
 *     - lapack::Uplo::Upper: Upper triangle of A is stored;
 *     - lapack::Uplo::Lower: Lower triangle of A is stored.
 *
 * @param[in] n
 *     The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *     The number of right hand sides, i.e., the number of columns
 *     of the matrix B. nrhs >= 0.
 *
 * @param[in] A
 *     The n-by-n matrix A, stored in an lda-by-n array.
 *     The triangular factor U or L from the Cholesky factorization
 *     $A = U^H U$ or $A = L L^H$, as computed by `lapack::potrs`.
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[in,out] B
 *     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
 *     On entry, the right hand side matrix B.
 *     On exit, the solution matrix X.
 *
 * @param[in] ldb
 *     The leading dimension of the array B. ldb >= max(1,n).
 *
 * @return = 0: successful exit
 *
 * @ingroup posv_computational
 */
template< typename TA, typename TB >
int potrs(
    Uplo uplo, idx_t n, idx_t nrhs,
    TA const* A, idx_t lda,
    TB* B, idx_t ldb )
{
    using blas::trsm;

    // constants
    TB one( 1.0 );

    // Check arguments
    lapack_error_if( uplo != Uplo::Upper && uplo != Uplo::Lower, -1 );
    lapack_error_if( n < 0, -2 );
    lapack_error_if( nrhs < 0, -3 );
    lapack_error_if( lda < n, -5 );
    lapack_error_if( ldb < n, -7 );

    // quick return
    if (n == 0 || nrhs == 0)
        return 0;

    if( uplo == Uplo::Upper ) {
        trsm(
            Layout::ColMajor, Side::Left, Uplo::Upper,
            Op::ConjTrans, Diag::NonUnit,
            n, nrhs, one, A, lda, B, ldb );
        trsm(
            Layout::ColMajor, Side::Left, Uplo::Upper,
            Op::NoTrans, Diag::NonUnit,
            n, nrhs, one, A, lda, B, ldb );
    }
    else {
        trsm(
            Layout::ColMajor, Side::Left, Uplo::Lower,
            Op::NoTrans, Diag::NonUnit,
            n, nrhs, one, A, lda, B, ldb );
        trsm(
            Layout::ColMajor, Side::Left, Uplo::Lower,
            Op::ConjTrans, Diag::NonUnit,
            n, nrhs, one, A, lda, B, ldb );
    }
    
    return 0;
}

} // lapack

#endif // __POTRS_HH__