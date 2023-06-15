union BatchAliasable {
    // ValueT s_slot_0[(BLOCK_THREADS) * BATCH_SIZE * BATCH_SIZE];
} batch_aliasable;

MergeItem merge_items[(TILE_ITEMS + 1 + ITEMS_PER_THREAD) * BATCH_SIZE];
MergeItem merge_items2[(TILE_ITEMS + 1 + ITEMS_PER_THREAD) * BATCH_SIZE];
