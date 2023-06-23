union BatchAliasable {
    ValueT s_slot_0[ BLOCK_THREADS * 9 ];
} batch_aliasable;
MergeItem merge_items_B_einsum [(TILE_ITEMS + 1 + ITEMS_PER_THREAD) *  1  * 3];
MergeItem merge_items_B_einsum_1 [(TILE_ITEMS + 1 + ITEMS_PER_THREAD) *  1  * 3];