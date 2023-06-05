struct _TempStorage
{
    CoordinateT tile_coords[2];
    ValueT input_buffer_0[(ITEMS_PER_THREAD + TILE_ITEMS + 1) * BATCH_SIZE];

    union Aliasable
    {
        struct {
            // Smem needed for tile of merge items
            MergeItem merge_items[(ITEMS_PER_THREAD + TILE_ITEMS + 1) * BATCH_SIZE];
        } batch_op;

        // Smem needed for block exchange
        typename BlockExchangeT::TempStorage exchange;

        // Smem needed for block-wide reduction
        typename BlockReduceT::TempStorage reduce;

        // Smem needed for tile scanning
        typename BlockScanT::TempStorage scan;

        // Smem needed for tile prefix sum
        typename BlockPrefixSumT::TempStorage prefix_sum;

    } aliasable;
};