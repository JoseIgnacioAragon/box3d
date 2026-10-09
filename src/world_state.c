// SPDX-License-Identifier: MIT

#include "core.h"
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

int b3World_SaveState( b3WorldId worldId, void* buffer, int capacity )
{
	b3World* world = b3GetUnlockedWorldFromId( worldId );
	if ( world == NULL )
	{
		return 0;
	}

	// The recording only holds the geometry registry the snapshot interns into.
	b3Recording* recording = b3CreateRecording( 0 );
	b3RecBuffer snapshot = { 0 };
	b3SerializeWorld( world, &snapshot, recording );

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
