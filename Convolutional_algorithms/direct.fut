module direct = {
  import "../modules/pad"

  let directConvolution [rows][cols]
                           (data:  [rows][cols]f32)
                           (kernel: [3][3]f32) (row: i32) (col: i32): f32 =

    unsafe
    let sum =
      data[row-1,col-1]*kernel[0,0] + data[row-1,col]*kernel[0,1] +
      data[row-1,col+1]*kernel[0,2] + data[row,col-1]*kernel[1,0] +
      data[row,  col]  *kernel[1,1] + data[row,col+1]*kernel[1,2] +
      data[row+1,col-1]*kernel[2,0] + data[row+1,col]*kernel[2,1] +
      data[row+1,col+1]*kernel[2,2]
    in sum
  
  let directConvolutionNoPad [rows][cols]
                           (data:  [rows][cols]f32)
                           (kernel: [3][3]f32) (row: i32) (col: i32): f32 =
    let sum = 
      (if (row != 0 && col != 0) then data[row-1,col-1] else 0) * kernel[0,0] + (if (row != 0) then data[row-1,col] else 0) * kernel[0,1] + 
      (if (row != 0 && col != cols-1) then data[row-1,col+1] else 0) * kernel[0,2] + (if (col != 0) then data[row,col-1] else 0) * kernel[1,0] +
      data[row, col] * kernel[1,1] + (if (row != cols-1) then data[row,col+1] else 0) * kernel[1,2] +
      (if (row != rows-1 && col != 0) then data[row+1,col-1] else 0) * kernel[2,0] + (if (row != rows-1) then data[row+1,col] else 0) * kernel[2,1] + 
      (if (row != rows-1 && col != cols-1) then data[row+1,col+1] else 0) * kernel[2,2]
    in sum


  let convolveData [rows][cols]
                      (data: [rows][cols]f32) (kernel: [3][3]f32): [][]f32 =
    unsafe
    map (\row ->
      map (\col ->
	          directConvolutionNoPad data kernel row col)
	        (1...cols-2))
        (1...rows-2)


  let main [rows][cols]
           (data: [rows][cols]f32) (kernel: [3][3]f32): [][]f32 =

    --let padded = pad.paddata data
    --let res = convolveData padded kernel
    let res = convolveData data kernel
    in res
}
-- ==
-- compiled input @ ../data/pyarray.in
-- output @ ../data/pyresult.in
