module mec = {
  import "../modules/pad"
  import "../modules/matmul"



  let interpretTile [rows][cols]
                    (tile: [rows][cols]f32): []f32 =
 
    let res = flatten tile
    in res
      
  
  let interpretDataFirst [rows][cols]
                    (data: [rows][cols]f32): [][]f32 =

    let res =
      unsafe
      map (\i ->   
            unsafe
            interpretTile data[0:(rows),i-1:i+2])
          (1...cols-2)
    in res

  let interpretDataSecond [rows][cols]
                          (data: [rows][cols]f32): [][]f32 =

    let Tdata = (transpose data)
    let res = 
      unsafe
      map (\i ->
        map (\j ->
              unsafe
             [if i == 0 || (j == 0 || j == (cols+1)) then 0 else Tdata[i-1, j-1], 
              if j == 0 || j == (cols+1) then 0 else Tdata[i, j-1], 
              if i == (rows-1) || (j == 0 || j == (cols+1)) then 0 else Tdata[i+1, j-1]
             ])
            (0...cols+1))
          (0...rows-1)

    in unflatten cols ((rows+2)*3) (flatten (flatten res))


  let convolvePartitions [rows][cols]
                   (i_data: [rows][cols]f32) (i_kernel: [9]f32): [][]f32 =

    let res = 
      unsafe 
      map (\row ->
        map (\col ->   
              unsafe
              --matmul.vecmul  row[col:col+9] i_data
              row[col]   * i_kernel[0] +
              row[col+1] * i_kernel[1] +
              row[col+2] * i_kernel[2] +
              row[col+3] * i_kernel[3] +
              row[col+4] * i_kernel[4] +
              row[col+5] * i_kernel[5] +
              row[col+6] * i_kernel[6] +
              row[col+7] * i_kernel[7] +
              row[col+8] * i_kernel[8])
            (range 0 (cols-6) 3))
           i_data
    in (transpose res)
  

  let main [rows_data][cols_data] [rows_kernel][cols_kernel]
           (data: [rows_data][cols_data]f32) (kernel: [rows_kernel][cols_kernel]f32): [][]f32 =

    let padded = pad.padImage data
    let i_data = interpretDataFirst padded
    --let i_data = interpretDataSecond data
    let i_kernel = interpretTile kernel
    let output = convolvePartitions i_data i_kernel
    in output
}