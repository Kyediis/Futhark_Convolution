import "pad"

let matmul1d [n]
           (x: [n]f32) (y: [n]f32): f32 =
  reduce (+) 0 (map2 (*) x y)

let interpretTile [rows][cols]
                    (tile: [rows][cols]f32): []f32 =
 
  let res = flatten tile
  in res
	    
  
let interpretData [rows][cols]
                  (data: [rows][cols]f32): [][9]f32 =
  let res = 
    unsafe 
    map (\i ->
  	     map (\j ->		
  	             unsafe
  	             interpretTile data[i-1:i+2,j-1:j+2])
         (1...cols-2))
      (1...rows-2)
  in flatten res
  

let convolveCols [rows]
                 (i_data: [rows][9]f32) (i_kernel: [9]f32): [][]f32 =
  
  let res = 
  	     map (\i ->		
  	             unsafe
  	             matmul1d i_data[i] i_kernel)
      (0...rows-1)
  in unflatten 4 4 res
  

let main [rows_data][cols_data] [rows_kernel][cols_kernel]
         (data: [rows_data][cols_data]f32) (kernel: [rows_kernel][cols_kernel]f32): [][]f32 =

  let padded = pad.padImage data
  let i_data = interpretData padded
  let i_kernel = interpretTile kernel
  let output = convolveCols i_data i_kernel
  in output

-- ==
--compiled input @ sharpness_filter.in
