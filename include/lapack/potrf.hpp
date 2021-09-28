/// @file potrf.hpp Computes the Cholesky factorization of a Hermitian positive definite matrix A
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/potrf.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __POTRF_HH__
#define __POTRF_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"
#include "lapack/lacgv.hpp"
#include "tblas.hpp"

namespace lapack {

/** Computes the Cholesky factorization of a Hermitian
 * positive definite matrix A.
 *
 * The factorization has the form
 *     $A = U^H U,$ if uplo = Upper, or
 *     $A = L L^H,$ if uplo = Lower,
 * where $U$ is an upper triangular matrix
 * and   $L$ is a  lower triangular matrix.
 *
 * This is the block version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *     - lapack::Uplo::Upper: Upper triangle of A is stored;
 *     - lapack::Uplo::Lower: Lower triangle of A is stored.
 *
 * @param[in] n
 *     The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *     The n-by-n matrix A, stored in an lda-by-n array.
 *     On entry, the Hermitian matrix A.
 *     - If uplo = Upper, the leading
 *     n-by-n upper triangular part of A contains the upper
 *     triangular part of the matrix A, and the strictly lower
 *     triangular part of A is not referenced.
 *
 *     - If uplo = Lower, the
 *     leading n-by-n lower triangular part of A contains the lower
 *     triangular part of the matrix A, and the strictly upper
 *     triangular part of A is not referenced.
 *
 *     - On successful exit, the factor U or L from the Cholesky
 *     factorization $A = U^H U$ or $A = L L^H.$
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= max(1,n).
 *
 * @return = 0: successful exit
 * @return > 0: if return value = i, the leading minor of order i is not
 *              positive definite, and the factorization could not be
 *              completed.
 *
 * @ingroup posv_computational
 */
template< typename T >
int potrf(
    Uplo uplo, idx_t n, T* A_, idx_t lda )
{
    typedef blas::real_type<T> real_t;
    using blas::isnan;
    using blas::sqrt;
    using blas::real;
    using blas::dot;
    using blas::gemv;
    using blas::scal;

    // Constants
    const real_t    rone( 1.0 );
    const T         one( 1.0 );
    const real_t    zero( 0.0 );

    // Check arguments
    lapack_error_if( uplo != Uplo::Upper && uplo != Uplo::Lower, -1 );
    lapack_error_if( n < 0, -2 );
    lapack_error_if( lda < n, -4 );

    // Quick return
    if (n == 0)
        return 0;

    // Matrix view
    #define A(i_, j_) A_[ (i_) + (j_)*lda ]
    
    if( uplo == Uplo::Upper ) {
        for (int_t j = 0; j < n-1; ++j) {
            real_t ajj = real( A(j,j) - dot(j, &A(0,j), 1, &A(0,j), 1) );
            if( ajj <= zero || isnan(ajj) ) {
                A(j,j) = ajj;
                return j+1;
            }
            ajj = sqrt(ajj);
            A(j,j) = ajj;
            
            lacgv(j, &A(0,j), 1);
            gemv(   
                Layout::ColMajor, Op::Trans, 
                j, n-j-1, -one, 
                &A(0,j+1), lda, 
                &A(0,j), 1, 
                one, &A(j,j+1), lda
            );
            lacgv(j, &A(0,j), 1);
            scal(n-j-1, rone/ajj, &A(j,j+1), lda);
        } {
            int_t j = n-1;
            real_t ajj = real( A(j,j) - dot(j, &A(0,j), 1, &A(0,j), 1) );
            if( ajj <= zero || isnan(ajj) ) {
                A(j,j) = ajj;
                return j+1;
            }
            A(j,j) = sqrt(ajj);
        }
    }
    else {
        for (int_t j = 0; j < n-1; ++j) {
            real_t ajj = real( A(j,j) - dot(j, &A(j,0), lda, &A(j,0), lda) );
            if( ajj <= zero || isnan(ajj) ) {
                A(j,j) = ajj;
                return j+1;
            }
            ajj = sqrt(ajj);
            A(j,j) = ajj;
            
            lacgv(j, &A(j,0), lda);
            gemv(   
                Layout::ColMajor, Op::NoTrans, 
                n-j-1, j, -one, 
                &A(j+1,0), lda, 
                &A(j,0), lda,
                one, &A(j+1,j), 1
            );
            lacgv(j, &A(j,0), lda);
            scal(n-j-1, rone/ajj, &A(j+1,j), 1);
        } {
            int_t j = n-1;
            real_t ajj = real( A(j,j) - dot(j, &A(j,0), lda, &A(j,0), lda) );
            if( ajj <= zero || isnan(ajj) ) {
                A(j,j) = ajj;
                return j+1;
            }
            A(j,j) = sqrt(ajj);
        }
    }

    #undef A
    return 0;
}

} // lapack

#endif // __POTRF_HH__