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

let interpretTile [rows][cols]
                    (tile: [rows][cols]f32): []f32 =
 
  let res = flatten tile
  in res
	    
  
let interpretData [rows][cols]
                  (data: [rows][cols]f32): [][]f32 =
  unsafe
  map (\i ->
  	flatten(map (\j ->		
  	       unsafe
  	       interpretTile data[i-1:i+2,j-1:j+2])
            (1...cols-2)))
      (1...rows-2)
  

-- let convolveTiles [rows][cols]
--                   (channel: [rows][cols]f32) (t_kernel: [4][4]f32)
-- 		  (h_tiles:i32) (v_tiles:i32): i32 =
  
--   map (\i ->
-- 	 map (\j ->
--                  unsafe		
-- 	         0)
--             0...1)
--       0...1
  

let main [rows_data][cols_data] [rows_kernel][cols_kernel]
         (data: [rows_data][cols_data]f32) (kernel: [rows_kernel][cols_kernel]f32): [][]f32 =

  let padded = pad.padImage data
  let i_data = interpretData padded
  let i_kernel = interpretTile kernel
  in i_data

-- ==
--compiled input @ filter.in
