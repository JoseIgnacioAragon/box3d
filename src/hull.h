// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "box3d/types.h"

#include <stddef.h>

// Big enough to handle AVX2.
#define B3_HULL_SOA_PADDING 8

static inline int b3GetHullSoaStride( int count )
{
	return ( count + B3_HULL_SOA_PADDING - 1 ) & ~( B3_HULL_SOA_PADDING - 1 );
}

uint64_t b3HashHullData( const b3HullData* hull );
bool b3CompareHullData( const b3HullData* hull1, const b3HullData* hull2 );

// Map keyed by hull content. The world hull database uses the value as a reference count,
// while compound baking stores uses the value as a byte offset. Implementation is in hull.c.
#define NAME b3HullMap
#define KEY_TY const b3HullData*
#define VAL_TY int
#define HEADER_MODE
#include "verstable.h"

// Total map allocation in bytes, excluding the stored hull data
size_t b3HullMapByteCount( b3HullMap* map );

// AXIOM CHANGE BEGIN
// Internal hull-aware entry points reuse existing packed coordinates; public query structs and saved geometry remain unchanged.
b3DistanceOutput b3ShapeDistanceWithHull( const b3DistanceInput* input, b3SimplexCache* cache, b3Simplex* simplexes, int simplexCapacity, const b3HullData* hull );
b3CastOutput b3ShapeCastWithHull( const b3ShapeCastPairInput* input, const b3HullData* hull );
// AXIOM CHANGE END
