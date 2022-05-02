// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_UTIL_HH
#define BLAS_UTIL_HH

#include <assert.h>

#include "base/utils.hpp"

/// Use to silence compiler warning of unused variable.
#define blas_unused( var ) ((void)var)

#define blas_error_if( cond ) tblas_error_if( cond )
#define blas_error_if_msg( cond, ... ) tblas_error_if_msg( cond, ... )

namespace blas {

    using namespace tlapack;

    // -----------------------------------------------------------------------------
    // Convert enum to LAPACK-style char.
    inline char layout2char( Layout layout ) { return char(layout); }
    inline char     op2char( Op     op     ) { return char(op);     }
    inline char   uplo2char( Uplo   uplo   ) { return char(uplo);   }
    inline char   diag2char( Diag   diag   ) { return char(diag);   }
    inline char   side2char( Side   side   ) { return char(side);   }

    // -----------------------------------------------------------------------------
    // Convert enum to LAPACK-style string.
    inline const char* layout2str( Layout layout )
    {
        switch (layout) {
            case Layout::ColMajor: return "col";
            case Layout::RowMajor: return "row";
            default:               return "";
        }
        return "";
    }

    inline const char* op2str( Op op )
    {
        switch (op) {
            case Op::NoTrans:   return "notrans";
            case Op::Trans:     return "trans";
            case Op::ConjTrans: return "conj";
            default:            return "";
        }
    }

    inline const char* uplo2str( Uplo uplo )
    {
        switch (uplo) {
            case Uplo::Lower:   return "lower";
            case Uplo::Upper:   return "upper";
            case Uplo::General: return "general";
        }
        return "";
    }

    inline const char* diag2str( Diag diag )
    {
        switch (diag) {
            case Diag::NonUnit: return "nonunit";
            case Diag::Unit:    return "unit";
        }
        return "";
    }

    inline const char* side2str( Side side )
    {
        switch (side) {
            case Side::Left:  return "left";
            case Side::Right: return "right";
        }
        return "";
    }

    // -----------------------------------------------------------------------------
    // Convert LAPACK-style char to enum.
    inline Layout char2layout( char layout )
    {
        layout = (char) toupper( layout );
        assert( layout == 'C' || layout == 'R' );
        return Layout( layout );
    }

    inline Op char2op( char op )
    {
        op = (char) toupper( op );
        assert( op == 'N' || op == 'T' || op == 'C' );
        return Op( op );
    }

    inline Uplo char2uplo( char uplo )
    {
        uplo = (char) toupper( uplo );
        assert( uplo == 'L' || uplo == 'U' || uplo == 'G' );
        return Uplo( uplo );
    }

    inline Diag char2diag( char diag )
    {
        diag = (char) toupper( diag );
        assert( diag == 'N' || diag == 'U' );
        return Diag( diag );
    }

    inline Side char2side( char side )
    {
        side = (char) toupper( side );
        assert( side == 'L' || side == 'R' );
        return Side( side );
    }

}

using blas::uplo2str;

#endif
