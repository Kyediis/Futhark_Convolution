module direct = {
  import "../modules/pad"

  let directConvolution [rows][cols]
                           (image:  [rows][cols]f32)
                           (kernel: [3][3]f32) (row: i32) (col: i32): f32 =

    unsafe
    let sum =
      image[row-1,col-1]*kernel[0,0] + image[row-1,col]*kernel[0,1] +
      image[row-1,col+1]*kernel[0,2] + image[row,col-1]*kernel[1,0] +
      image[row,  col]  *kernel[1,1] + image[row,col+1]*kernel[1,2] +
      image[row+1,col-1]*kernel[2,0] + image[row+1,col]*kernel[2,1] +
      image[row+1,col+1]*kernel[2,2]
    in sum


  let convolveData [rows][cols]
                      (data: [rows][cols]f32) (kernel: [3][3]f32): [][]f32 =
    unsafe
    map (\row ->
      map (\col ->
	          directConvolution data kernel row col)
	        (1...cols-2))
        (1...rows-2)


  let main [rows][cols]
           (data: [rows][cols]f32) (kernel: [3][3]f32): [][]f32 =

    let padded = pad.padImage data
    let res = convolveData padded kernel
    in res
}
-- ==
-- compiled input @ ../data/pyarray.in
-- output @ ../data/pyresult.in
