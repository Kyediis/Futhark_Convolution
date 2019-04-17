import "pad"

let matmul1d [n]
           (x: [n]f32) (y: [n]f32): f32 =
  reduce (+) 0 (map2 (*) x y)

let interpretTile [rows][cols]
                    (tile: [rows][cols]f32): []f32 =
 
  let res = flatten tile
  in res
      
  
let interpretData [rows][cols]
                  (data: [rows][cols]f32): [][]f32 =
  let res =
      unsafe
      map (\i ->   
              unsafe
              interpretTile data[0:(rows),i-1:i+2])
          (1...cols-2)
  in res

let partitionData [rows][cols]
                  (i_data: [rows][cols]f32): [][][]f32 =
  let res =
      unsafe
      map (\i ->   
              unsafe
              i_data[0:(rows),i:i+9])
          (range 0 ((cols/2)+3) 3)
  in res
  

let convolvePartitions [partitions][rows][cols]
                 (p_data: [partitions][rows][cols]f32) (i_kernel: [9]f32): [][]f32 =
  let res = 
    unsafe 
    map (\i ->
         map (\j ->   
                 unsafe
                 matmul1d p_data[i,j] i_kernel)
         (0...rows-1))
      (0...partitions-1)
  in res
  

let main [rows_data][cols_data] [rows_kernel][cols_kernel]
         (data: [rows_data][cols_data]f32) (kernel: [rows_kernel][cols_kernel]f32): [][]f32 =

  let padded = pad.padImage data
  let i_kernel = interpretTile kernel
  let i_data = interpretData padded
  let p_data = partitionData i_data
  let output = convolvePartitions p_data i_kernel
  in output

-- ==
--compiled input @ sharpness_filter.in