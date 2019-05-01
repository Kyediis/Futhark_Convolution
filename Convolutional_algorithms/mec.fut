module mec = {
  import "../modules/pad"

  let matmul1d [n]
               (x: [n]f32) (y: [n]f32): f32 =

    reduce (+) 0 (map2 (*) x y)

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
              [Tdata[i-1, j], Tdata[i, j], Tdata[i+1, j]])
            (0...cols-1))
          (1...rows-2)

    in unflatten (cols-2) (rows*3) (flatten (flatten res))


  let partitionDataFirst [rows][cols]
                  (i_data: [rows][cols]f32): [][][]f32 =

    let res =
      unsafe
      map (\i ->   
            unsafe
            i_data[0:(rows),i:i+9])
          (range 0 ((cols-9)+3) 3)
    in res

  let partitionDataSecond [rows][cols]
                  (i_data: [rows][cols]f32): [][][]f32 =

    let res = 
      unsafe
      map (\p ->
        map (\i ->
          map (\j ->
                unsafe
                i_data[i,j+p])
              (0...9-1))
            (0...rows-1))
          (range 0 (rows*3) 3)

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
    let i_data = interpretDataSecond padded
    let p_data = partitionDataSecond i_data
    let output = convolvePartitions p_data i_kernel
    in output
}
-- ==
-- compiled input @ ../data/pyarray.txt
-- output @ ../data/pyresult.txt