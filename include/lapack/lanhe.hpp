/// @file lanhe.hpp Returns the norm of a Hermitian matrix.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lanhe.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LANHE_HH__
#define __LANHE_HH__

#include "lapack/types.hpp"
#include "lapack/lassq.hpp"

namespace lapack {

/** Returns the value of the one norm, Frobenius norm,
 * infinity norm, or the element of largest absolute value of a
 * complex hermitian matrix A.
 *
 * For real matrices, this is an alias for `lapack::lansy`.
 * For complex symmetric matrices, see `lapack::lansy`.
 *
 * @param[in] norm
 *     The value to be returned:
 *     - lapack::Norm::Max: max norm: max(abs(A(i,j))).
 *                          Note this is not a consistent matrix norm.
 *     - lapack::Norm::One: one norm: maximum column sum
 *     - lapack::Norm::Inf: infinity norm: maximum row sum
 *     - lapack::Norm::Fro: Frobenius norm: square root of sum of squares
 *
 * @param[in] uplo
 *     Whether the upper or lower triangular part of the
 *     hermitian matrix A is to be referenced.
 *     - lapack::Uplo::Upper: Upper triangular part of A is referenced
 *     - lapack::Uplo::Lower: Lower triangular part of A is referenced
 *
 * @param[in] n
 *     The order of the matrix A. n >= 0. When n = 0, returns zero.
 *
 * @param[in] A
 *     The n-by-n matrix A, stored in an lda-by-n array.
 *     The hermitian matrix A.
 *     - If uplo = Upper, the leading n-by-n
 *     upper triangular part of A contains the upper triangular part
 *     of the matrix A, and the strictly lower triangular part of A
 *     is not referenced.
 *
 *     - If uplo = Lower, the leading n-by-n lower
 *     triangular part of A contains the lower triangular part of
 *     the matrix A, and the strictly upper triangular part of A is
 *     not referenced.
 *
 *     - Note that the imaginary parts of the diagonal
 *     elements need not be set and are assumed to be zero.
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= max(n,1).
 *
 * @ingroup norm
 */
template <typename TA>
real_type<TA> lanhe(
    Norm normType, Uplo uplo, blas::idx_t n,
    const TA *A_, blas::idx_t lda )
{
    typedef real_type<TA> real_t;
    using blas::isnan;
    using blas::sqrt;
    using blas::real;
    using blas::pow;
    
    // constants
    const real_t zero(0.0);
    const real_t safe_max = pow(
        std::numeric_limits<real_t>::radix,
        std::numeric_limits<real_t>::max_exponent - real_t(1.0) );

    // quick return
    if (n == 0)
        return zero;

    // Matrix views
    #define A(i_, j_) A_[ (i_) + (j_)*lda ]

    // Norm value
    real_t norm(0.0);

    if( normType == Norm::Max )
    {
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i <= j; ++i)
                {
                    real_t temp = blas::abs( A(i,j) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = j; i < n; ++i)
                {
                    real_t temp = blas::abs( A(i,j) );

                    if (temp > norm)
                        norm = temp;
                    else {
                        if ( isnan(temp) ) 
                            return temp;
                    }
                }
            }
        }
    }
    else if ( normType == Norm::One || normType == Norm::Inf )
    {
        real_t *work = new real_t[n];
        for (idx_t i = 0; i < n; ++i)
            work[i] = zero;
        
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; ++j) {   
                real_t sum = zero;
                for (idx_t i = 0; i < j; ++i) {
                    real_t temp = blas::abs( A(i,j) );
                    sum += temp;
                    work[i] += temp;
                }
                work[j] = sum + blas::abs( real(A(j,j)) );
            }
            for (idx_t i = 0; i < n; ++i)
            {
                const real_t& sum = work[i];

                if (sum > norm)
                    norm = sum;
                else {
                    if (isnan(sum)) {
                        delete[] work;
                        return sum;
                    }
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j) {   
                real_t sum = work[j] + blas::abs( real(A(j,j)) );
                for (idx_t i = j+1; i < n; ++i) {
                    real_t temp = blas::abs( A(i,j) );
                    sum += temp;
                    work[i] += temp;
                }
                if (sum > norm)
                    norm = sum;
                else {
                    if (isnan(sum)) {
                        delete[] work;
                        return sum;
                    }
                }
            }
        }
        delete[] work;
    }
    else if ( normType == Norm::Fro )
    {
        real_t scale(0.0), sum(1.0);
        // Sum all elements from one side out of the main diagonal
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 1; j < n; ++j)
                lassq(j, &(A(0,j)), 1, scale, sum);
        }
        else {
            for (idx_t j = 0; j < n-1; ++j)
                lassq(n-j-1, &(A(j+1,j)), 1, scale, sum);
        }
        // Multiplies the sum by 2
        if( sum < safe_max ) {
            sum *= 2;
        } else {
            scale *= sqrt(2);
        }
        // Sum the elements in the main diagonal
        lassq(n-1, &(A(0,0)), lda+1, scale, sum,
            []( const TA& x ){ return blas::abs(real(x)); }
        );
        // Compute the norm
        norm = scale * sqrt(sum);
    }

    #undef A
    return norm;
}

} // lapack

#endif // __LANHE_HH__