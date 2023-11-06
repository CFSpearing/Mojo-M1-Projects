from memory import stack_allocation
from algorithm import vectorize_unroll


alias type = DType.float32
alias nelts = simdwidthof[DType.float32]()

fn tile_parallel[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    @parameter
    fn row(yo: Int):
        let y = tile_y * yo
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

    parallelize[row](end_y // tile_y, 512)

struct Matrix:
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    # Initialize zeroeing all values
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[DType.float32]):
        self.data = data
        self.rows = rows
        self.cols = cols

    ## Initialize with random values
    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

fn accumulate_registers(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
        # Allocate the tile of accumulators on the stack.
        var accumulators = Matrix(tile_i, tile_j, stack_allocation[tile_i * tile_j, DType.float32]())

        for k in range(0, A.cols):
            @parameter
            fn calc_tile_row[i: Int]():
                @parameter
                fn calc_tile_cols[nelts: Int](j: Int):
                    accumulators.store[nelts](i, j, accumulators.load[nelts](i, j) + A[io + i, k] * B.load[nelts](k, jo + j))

                vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

            unroll[tile_i, calc_tile_row]()

        # Copy the local tile to the output
        for i in range(tile_i):
            for j in range(tile_j):
                C[io + i, jo + j] = accumulators[i, j]

    alias tile_i = 4
    alias tile_j = nelts*4
    tile_parallel[calc_tile, tile_j, tile_i](C.cols, C.rows)