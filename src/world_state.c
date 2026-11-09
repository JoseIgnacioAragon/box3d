// SPDX-License-Identifier: MIT

#include "core.h"
#include "body.h"
#include "contact.h"
#include "island.h"
#include "joint.h"
#include "sensor.h"
#include "shape.h"
#include "solver_set.h"
#include "physics_world.h"
#include "recording.h"
#include "replay.h"
#include "world_snapshot.h"

#include "box3d/box3d.h"

#include <string.h>

// b3World_SaveState image, little-endian:
//   b3StateHeader
//   b3StateGeometry, geometryCount of them
//   the world snapshot (world_snapshot.c), at snapshotOffset
//   each geometry's bytes, at its offset
// Offsets are from the image's start, aligned to B3_STATE_ALIGN. A snapshot's geometry id is
// the geometry's index in the table.

#define B3_STATE_MAGIC 0x53573342u // 'B3WS'
#define B3_STATE_VERSION 1u
#define B3_STATE_ALIGN 16

typedef struct b3StateHeader
{
	uint32_t magic;
	uint32_t version;
	int32_t geometryCount;
	int32_t snapshotOffset;
	int32_t snapshotSize;
	int32_t padding;
} b3StateHeader;

typedef struct b3StateGeometry
{
	int32_t kind;
	int32_t byteCount;
	int32_t offset;
	int32_t padding;
} b3StateGeometry;

static int b3AlignStateOffset( int offset )
{
	return ( offset + B3_STATE_ALIGN - 1 ) & ~( B3_STATE_ALIGN - 1 );
}

static int b3SaveWorldStateImage( b3World* world, const b3Shape* shapeImages, void* buffer, int capacity )
{
	if ( world == NULL )
	{
		return 0;
	}

	// The recording only holds the geometry registry the snapshot interns into.
	b3Recording* recording = b3CreateRecording( 0 );
	b3RecBuffer snapshot = { 0 };
	b3SerializeCapturedWorld( world, shapeImages, &snapshot, recording );

	int geometryCount = recording->registry.entries.count;
	int tableOffset = b3AlignStateOffset( (int)sizeof( b3StateHeader ) );
	int snapshotOffset = b3AlignStateOffset( tableOffset + geometryCount * (int)sizeof( b3StateGeometry ) );
	int size = b3AlignStateOffset( snapshotOffset + snapshot.size );
	for ( int i = 0; i < geometryCount; ++i )
	{
		size = b3AlignStateOffset( size + recording->registry.entries.data[i].byteCount );
	}

	if ( buffer != NULL && size <= capacity )
	{
		uint8_t* bytes = (uint8_t*)buffer;
		memset( bytes, 0, (size_t)size );

		b3StateHeader header = {
			.magic = B3_STATE_MAGIC,
			.version = B3_STATE_VERSION,
			.geometryCount = geometryCount,
			.snapshotOffset = snapshotOffset,
			.snapshotSize = snapshot.size,
		};
		memcpy( bytes, &header, sizeof( header ) );
		memcpy( bytes + snapshotOffset, snapshot.data, (size_t)snapshot.size );

		int offset = b3AlignStateOffset( snapshotOffset + snapshot.size );
		for ( int i = 0; i < geometryCount; ++i )
		{
			b3GeometryEntry* entry = recording->registry.entries.data + i;
			b3StateGeometry geometry = {
				.kind = (int32_t)entry->kind,
				.byteCount = entry->byteCount,
				.offset = offset,
			};
			memcpy( bytes + tableOffset + i * (int)sizeof( b3StateGeometry ), &geometry, sizeof( geometry ) );
			memcpy( bytes + offset, entry->bytes, (size_t)entry->byteCount );
			offset = b3AlignStateOffset( offset + entry->byteCount );
		}
	}

	b3RecBufFree( &snapshot );
	b3DestroyRecording( recording );
	return size;
}

int b3World_SaveState( b3WorldId worldId, void* buffer, int capacity )
{
	return b3SaveWorldStateImage( b3GetUnlockedWorldFromId( worldId ), NULL, buffer, capacity );
}

struct b3StateCapture
{
	b3World world;
	b3Shape* shapeImages;
	uint8_t* bytes;
	size_t capacity;
	size_t used;
	bool valid;
};

// Walk the snapshot serializer's read set twice: count, then copy into one reusable allocation.
// This is a serialization view, not a registered world, and must never be stepped or destroyed as one.
static void* b3CaptureStateBytes( b3StateCapture* capture, const void* source, size_t size )
{
	size_t start = ( capture->used + 63 ) & ~(size_t)63;
	capture->used = start + size;
	if ( capture->bytes == NULL || size == 0 )
	{
		return NULL;
	}
	void* result = capture->bytes + start;
	memcpy( result, source, size );
	return result;
}

static void b3CaptureWorldReadSet( b3StateCapture* capture, const b3World* source )
{
	b3World* copy = &capture->world;
	*copy = *source;
	capture->used = 0;
	bool writing = capture->bytes != NULL;
#define B3_CAPTURE_ARRAY( dst, src ) \
	do { void* p = b3CaptureStateBytes( capture, (src).data, (size_t)(src).count * sizeof( *(src).data ) ); \
	if ( writing ) { (dst).data = p; } } while ( 0 )
#define B3_CAPTURE_POINTER( dst, src, count ) \
	do { void* p = b3CaptureStateBytes( capture, (src), (size_t)(count) * sizeof( *(src) ) ); \
	if ( writing ) { (dst) = p; } } while ( 0 )
	B3_CAPTURE_ARRAY( copy->bodyIdPool.freeArray, source->bodyIdPool.freeArray );
	B3_CAPTURE_ARRAY( copy->shapeIdPool.freeArray, source->shapeIdPool.freeArray );
	B3_CAPTURE_ARRAY( copy->contactIdPool.freeArray, source->contactIdPool.freeArray );
	B3_CAPTURE_ARRAY( copy->jointIdPool.freeArray, source->jointIdPool.freeArray );
	B3_CAPTURE_ARRAY( copy->islandIdPool.freeArray, source->islandIdPool.freeArray );
	B3_CAPTURE_ARRAY( copy->solverSetIdPool.freeArray, source->solverSetIdPool.freeArray );
	B3_CAPTURE_ARRAY( copy->solverSets, source->solverSets );
	for ( int i = 0; i < source->solverSets.count; ++i )
	{
		const b3SolverSet* s = source->solverSets.data + i;
		b3SolverSet scratch = *s;
		b3SolverSet* d = writing ? copy->solverSets.data + i : &scratch;
		B3_CAPTURE_ARRAY( d->bodySims, s->bodySims );
		B3_CAPTURE_ARRAY( d->bodyStates, s->bodyStates );
		B3_CAPTURE_ARRAY( d->jointSims, s->jointSims );
		B3_CAPTURE_ARRAY( d->contactIndices, s->contactIndices );
		B3_CAPTURE_ARRAY( d->islandSims, s->islandSims );
	}
	B3_CAPTURE_ARRAY( copy->bodies, source->bodies );
	B3_CAPTURE_ARRAY( copy->shapes, source->shapes );
	capture->shapeImages = b3CaptureStateBytes( capture, source->shapes.data,
		(size_t)source->shapes.count * sizeof( b3Shape ) );
	for ( int i = 0; i < source->shapes.count; ++i )
	{
		const b3Shape* s = source->shapes.data + i;
		if ( s->id != i ) { continue; }
		b3Shape scratch = *s;
		b3Shape* d = writing ? copy->shapes.data + i : &scratch;
		if ( s->materials != NULL )
		{
			B3_CAPTURE_POINTER( d->materials, s->materials, s->materialCount );
		}
		switch ( s->type )
		{
			case b3_hullShape:
				d->hull = b3CaptureStateBytes( capture, s->hull, s->hull->byteCount ); break;
			case b3_meshShape:
				d->mesh.data = b3CaptureStateBytes( capture, s->mesh.data, s->mesh.data->byteCount ); break;
			case b3_heightShape:
				d->heightField = b3CaptureStateBytes( capture, s->heightField, s->heightField->byteCount ); break;
			case b3_compoundShape:
			{
				uint8_t* bytes = b3CaptureStateBytes( capture, s->compound, s->compound->byteCount );
				if ( writing ) { d->compound = b3ConvertBytesToCompound( bytes, s->compound->byteCount ); }
				break;
			}
			default: break; // sphere and capsule geometry is inline
		}
	}
	B3_CAPTURE_ARRAY( copy->fatAABBs, source->fatAABBs );
	B3_CAPTURE_ARRAY( copy->contacts, source->contacts );
	for ( int i = 0; i < source->contacts.count; ++i )
	{
		const b3Contact* s = source->contacts.data + i;
		if ( s->contactId != i ) { continue; }
		b3Contact scratch = *s;
		b3Contact* d = writing ? copy->contacts.data + i : &scratch;
		B3_CAPTURE_POINTER( d->manifolds, s->manifolds, s->manifoldCount );
		if ( s->flags & b3_simMeshContact )
		{
			B3_CAPTURE_ARRAY( d->meshContact.triangleCache, s->meshContact.triangleCache );
		}
	}
	B3_CAPTURE_ARRAY( copy->joints, source->joints );
	B3_CAPTURE_ARRAY( copy->sensors, source->sensors );
	for ( int i = 0; i < source->sensors.count; ++i )
	{
		const b3Sensor* s = source->sensors.data + i;
		b3Sensor scratch = *s;
		b3Sensor* d = writing ? copy->sensors.data + i : &scratch;
		B3_CAPTURE_ARRAY( d->hits, s->hits );
		B3_CAPTURE_ARRAY( d->overlaps1, s->overlaps1 );
		B3_CAPTURE_ARRAY( d->overlaps2, s->overlaps2 );
	}
	B3_CAPTURE_ARRAY( copy->islands, source->islands );
	for ( int i = 0; i < source->islands.count; ++i )
	{
		const b3Island* s = source->islands.data + i;
		b3Island scratch = *s;
		b3Island* d = writing ? copy->islands.data + i : &scratch;
		B3_CAPTURE_ARRAY( d->bodies, s->bodies );
		B3_CAPTURE_ARRAY( d->contacts, s->contacts );
		B3_CAPTURE_ARRAY( d->joints, s->joints );
	}
	for ( int i = 0; i < b3_bodyTypeCount; ++i )
	{
		const b3DynamicTree* s = &source->broadPhase.trees[i];
		b3DynamicTree* d = &copy->broadPhase.trees[i];
		B3_CAPTURE_POINTER( d->nodes, s->nodes, s->nodeEnd );
		B3_CAPTURE_POINTER( d->parents, s->parents, s->nodeEnd );
		B3_CAPTURE_POINTER( d->proxies, s->proxies, s->proxyCapacity );
	}
	B3_CAPTURE_POINTER( copy->broadPhase.pairSet.items, source->broadPhase.pairSet.items, source->broadPhase.pairSet.capacity );
	for ( int i = 0; i < B3_GRAPH_COLOR_COUNT; ++i )
	{
		const b3GraphColor* s = &source->constraintGraph.colors[i];
		b3GraphColor* d = &copy->constraintGraph.colors[i];
		if ( i != B3_OVERFLOW_INDEX )
		{
			B3_CAPTURE_POINTER( d->bodySet.bits, s->bodySet.bits, s->bodySet.blockCount );
		}
		B3_CAPTURE_ARRAY( d->jointSims, s->jointSims );
		B3_CAPTURE_ARRAY( d->convexContacts, s->convexContacts );
		B3_CAPTURE_ARRAY( d->contacts, s->contacts );
	}
	B3_CAPTURE_ARRAY( copy->names.entries, source->names.entries );
	for ( int i = 0; i < source->names.entries.count; ++i )
	{
		const b3NameEntry* s = source->names.entries.data + i;
		char* name = b3CaptureStateBytes( capture, s->name, s->length );
		if ( writing ) { copy->names.entries.data[i].name = name; }
	}
#undef B3_CAPTURE_ARRAY
#undef B3_CAPTURE_POINTER
}

b3StateCapture* b3CreateStateCapture( void )
{
	b3StateCapture* capture = b3Alloc( sizeof( b3StateCapture ) );
	memset( capture, 0, sizeof( *capture ) );
	return capture;
}

bool b3World_CaptureState( b3WorldId worldId, b3StateCapture* capture )
{
	b3World* world = b3GetUnlockedWorldFromId( worldId );
	if ( world == NULL || capture == NULL ) { return false; }
	capture->valid = false;
	uint8_t* previous = capture->bytes;
	capture->bytes = NULL;
	b3CaptureWorldReadSet( capture, world );
	if ( capture->used > capture->capacity )
	{
		b3Free( previous, capture->capacity );
		capture->capacity = ( capture->used + 1048575 ) & ~(size_t)1048575;
		previous = b3Alloc( capture->capacity );
	}
	capture->bytes = previous;
	if ( capture->bytes == NULL ) { return false; }
	b3CaptureWorldReadSet( capture, world );
	capture->valid = true;
	return true;
}

int b3SaveCapturedState( const b3StateCapture* capture, void* buffer, int capacity )
{
	if ( capture == NULL || !capture->valid ) { return 0; }
	return b3SaveWorldStateImage( (b3World*)&capture->world, capture->shapeImages, buffer, capacity );
}

void b3DestroyStateCapture( b3StateCapture* capture )
{
	if ( capture == NULL ) { return; }
	b3Free( capture->bytes, capture->capacity );
	b3Free( capture, sizeof( *capture ) );
}

bool b3World_RestoreState( b3WorldId worldId, const void* data, int size )
{
	b3World* world = b3GetUnlockedWorldFromId( worldId );
	if ( world == NULL || data == NULL || size < (int)sizeof( b3StateHeader ) )
	{
		return false;
	}

	const uint8_t* bytes = (const uint8_t*)data;
	b3StateHeader header;
	memcpy( &header, bytes, sizeof( header ) );
	int tableOffset = b3AlignStateOffset( (int)sizeof( b3StateHeader ) );
	if ( header.magic != B3_STATE_MAGIC || header.version != B3_STATE_VERSION || header.geometryCount < 0 ||
		 header.geometryCount > ( size - tableOffset ) / (int)sizeof( b3StateGeometry ) || header.snapshotOffset < tableOffset ||
		 header.snapshotSize < 0 || header.snapshotOffset > size - header.snapshotSize )
	{
		return false;
	}

	int geometryCount = header.geometryCount;
	b3RegistrySlot* slots = NULL;
	if ( geometryCount > 0 )
	{
		slots = (b3RegistrySlot*)b3Alloc( (size_t)geometryCount * sizeof( b3RegistrySlot ) );
	}
	bool ok = true;
	for ( int i = 0; i < geometryCount; ++i )
	{
		b3StateGeometry geometry;
		memcpy( &geometry, bytes + tableOffset + i * (int)sizeof( b3StateGeometry ), sizeof( geometry ) );
		bool known = geometry.kind == b3_geometryHull || geometry.kind == b3_geometryMesh || geometry.kind == b3_geometryHeightField;
		if ( known == false || geometry.byteCount < 0 || geometry.offset < 0 || geometry.offset > size - geometry.byteCount )
		{
			ok = false;
			break;
		}
		slots[i] = (b3RegistrySlot){
			.kind = (b3GeometryKind)geometry.kind,
			.byteCount = geometry.byteCount,
			.bytes = (uint8_t*)( bytes + geometry.offset ),
			.live = NULL,
		};
	}

	if ( ok )
	{
		b3RecReader reader;
		memset( &reader, 0, sizeof( reader ) );
		reader.data = bytes;
		reader.size = size;
		reader.ok = true;
		reader.replayWorldId = worldId;
		reader.slots = slots;
		reader.slotCount = geometryCount;
		ok = b3DeserializeIntoShell( bytes + header.snapshotOffset, header.snapshotSize, world, &reader );
	}

	if ( slots != NULL )
	{
		b3Free( slots, (size_t)geometryCount * sizeof( b3RegistrySlot ) );
	}
	return ok;
}
