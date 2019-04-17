import "pad"
		   
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


let convolveChannel [rows1][cols1] [rows2][cols2]
                    (channel: [rows1][cols1]f32) (kernel: [rows2][cols2]f32): [][]f32 =

  map (\row ->
    map (\col ->
	        directConvolution channel kernel row col)
	      (1...cols1-2))
      (1...rows1-2)


let main [rows1][cols1] [rows2][cols2]
         (image: [rows1][cols1]f32) (kernel: [rows2][cols2]f32): [][]f32 =

  let padded = pad.padImage image
  let res = convolveChannel padded kernel
  in res
	  
-- ==
--compiled input @ filter.in
