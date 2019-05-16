module im2col = {
  import "../modules/pad"
  let matmul1d [n]
               (x: [n]f32) (y: [n]f32): f32 =

    reduce (+) 0 (map2 (*) x y)

  let interpretTile [rows][cols]
                    (tile: [rows][cols]f32): []f32 =

    let res = flatten tile
    in res
      
  
  let interpretDataFirst [rows][cols]
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
  
  let interpretDataSecond [rows][cols]
                          (data: [rows][cols]f32): [][9]f32 =

    let res = 
      unsafe
      map (\flat ->
             unsafe
             let i = 1+flat / (cols-2)
             let j = 1+flat % (cols-2)
             in [data[i-1,j-1], data[i-1, j], data[i-1, j+1],
                 data[i,j-1],   data[i, j],   data[i, j+1],
                 data[i+1,j-1], data[i+1, j], data[i+1, j+1]
                ])
          (iota ((rows-2) * (cols-2)))
    in res

  let noPadInterpret [rows][cols]
                          (data: [rows][cols]f32): [][9]f32 =

    let res = 
      unsafe
      map (\flat ->
             unsafe
             let i = flat / (cols)
             let j = flat % (cols)
             in [if (i != 0 && j != 0) then data[i-1,j-1] else 0, if (i != 0) then data[i-1,j] else 0, if (i != 0 && j != cols-1) then data[i-1,j+1] else 0,
                 if (j != 0) then data[i,j-1] else 0,                              data[i, j],         if (j!= cols-1) then data[i,j+1] else 0,
                 if (i != rows-1 && j != 0) then data[i+1,j-1] else 0, if (i != rows-1) then data[i+1,j] else 0, if (i != rows-1 && j != cols-1) then data[i+1,j+1] else 0
                ])
          (iota ((rows) * (cols)))
    in res

  let convolveCols [rows]
                   (i_data: [rows][9]f32) (i_kernel: [9]f32)
                   (output_rows:i32) (output_cols:i32) : [][]f32 = 

    let res = 
      map (\row ->   
            unsafe
            --matmul1d row i_kernel
            row[0] * i_kernel[0] +
            row[1] * i_kernel[1] +
            row[2] * i_kernel[2] +
            row[3] * i_kernel[3] +
            row[4] * i_kernel[4] +
            row[5] * i_kernel[5] +
            row[6] * i_kernel[6] +
            row[7] * i_kernel[7] +
            row[8] * i_kernel[8]
            )
          i_data
    in unflatten output_rows output_cols res
  

  let main [rows_data][cols_data] [rows_kernel][cols_kernel]
           (data: [rows_data][cols_data]f32) (kernel: [rows_kernel][cols_kernel]f32): [][]f32 =

    let rows = rows_data
    let cols = cols_data
    --let padded = pad.padImage data
    --let i_data = interpretDataSecond padded
    let i_data = noPadInterpret data
    let i_kernel = interpretTile kernel
    let output = convolveCols i_data i_kernel rows cols
    in output
}
-- ==
-- compiled input @ ../data/pyarray.txt
-- output @ ../data/pyresult.txt