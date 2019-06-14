module matmul = {
  let matmul [n][m][p]
           (x: [n][m]f32) (y: [m][p]f32): [n][p]f32 =
  
    map (\xr ->
	    map (\yc ->
	          reduce (+) 0 (map2 (*) xr yc))
               (transpose y))
        x

  let vecmul [n]
               (x: [n]f32) (y: [n]f32): f32 =

    reduce (+) 0 (map2 (*) x y)
}