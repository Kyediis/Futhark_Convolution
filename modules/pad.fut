module pad = {
  let padImage [rows][cols]
               (image: [rows][cols]f32): [][]f32 =
    map (\row ->
      map (\col ->
              if row > 0 && row < rows+1 && col > 0 && col < cols+1
              then
		            unsafe
		            image[row-1, col-1]		    
              else
		            unsafe
		            0)
          (0...cols+1))
        (0...rows+1)
}