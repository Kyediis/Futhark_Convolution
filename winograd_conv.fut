import "pad"

let matmul [n][m][p]
           (x: [n][m]f32) (y: [m][p]f32): [n][p]f32 =
  map (\xr ->
	 map (\yc ->
	       reduce (+) 0 (map2 (*) xr yc))
             (transpose y))
      x


let pointwise [n][m]
              (x: [n][m]f32) (y: [n][m]f32): [n][m]f32 =
  map (\r ->
	map (\c ->
	       x[r,c] * y[r,c])
            (0...m-1))
      (0...n-1)
      

let transformKernel (kernel: [3][3]f32): [4][4]f32 =
  let G:[][]f32  = [[1.0,0.0,0.0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0.0,0.0,1.0]]--kernel
  let res = matmul (matmul G kernel) (transpose G)
  in res

  
let winogradConvolution [rows][cols]
		      (tile:  [rows][cols]f32)
		      (t_kernel: [4][4]f32): [2][2]f32 =
  
  let BT:[][]f32 = [[1.0,0.0,-1.0,0.0],[0.0,1.0,1.0,0.0],[0.0,-1.0,1.0,0.0],[0.0,1.0,0.0,-1.0]]--data
  let AT:[][]f32 = [[1.0,1.0,1.0,0.0],[0.0,1.0,-1.0,-1.0]]--output
		   
  let t_data = matmul (matmul BT tile) (transpose BT)
  let t_mult = pointwise t_data t_kernel 
  let res = matmul(matmul AT t_mult) (transpose AT)
  in res
  

let convolveTiles [rows][cols]
                  (channel: [rows][cols]f32) (t_kernel: [4][4]f32)
		  (h_tiles:i32) (v_tiles:i32) : [][]f32 =
  
  map (\i ->
	 flatten (transpose (map (\j ->
            unsafe		
	    flatten (winogradConvolution channel[i:i+4,j:j+4] t_kernel))
           (range 0 (h_tiles*2) 2))))
      (range 0 (v_tiles*2) 2)
  

let main [rows_data][cols_data] [rows_kernel][cols_kernel]
         (image: [rows_data][cols_data]f32) (kernel: [rows_kernel][cols_kernel]f32): [][]f32 =

  let padded = pad.padImage image
  let horizontal_tiles = cols_data / 2
  let vertical_tiles = rows_data / 2
  let t_kernel = transformKernel kernel
  
  let res =
    if (((rows_data*cols_data) % 4) == 0) then
      convolveTiles padded t_kernel horizontal_tiles vertical_tiles
    else
    [[2]]
  in res

-- ==
--compiled input @ filter.in
