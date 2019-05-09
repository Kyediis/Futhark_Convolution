module winograd = {
  import "../modules/pad"
  import "../modules/matmul"

  let pointwise [n][m]
              (x: [n][m]f32) (y: [n][m]f32): [n][m]f32 =
    map (\r ->
	    map (\c ->
	          x[r,c] * y[r,c])
          (0...m-1))
        (0...n-1)
      

  let transformKernel (kernel: [3][3]f32): [4][4]f32 =

    --let G:[][]f32  = [[1.0,0.0,0.0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0.0,0.0,1.0]]--kernel

    let kernel2 = [[kernel[0,0], kernel[0,1], kernel[0,2]],
    	          [(kernel[0,0]+kernel[1,0]+kernel[2,0])/2, (kernel[0,1]+kernel[1,1]+kernel[2,1])/2,  (kernel[0,2]+kernel[1,2]+kernel[2,2])/2],
    		      [(kernel[0,0]-kernel[1,0]+kernel[2,0])/2, (kernel[0,1]-kernel[1,1]+kernel[2,1])/2,  (kernel[0,2]-kernel[1,2]+kernel[2,2])/2],
    		      [kernel[2,0], kernel[2,1], kernel[2,2]]]

    let kernel3 = [[kernel2[0,0], (kernel2[0,0]+kernel2[0,1]+kernel2[0,2])/2, (kernel2[0,0]-kernel2[0,1]+kernel2[0,2])/2, kernel2[0,2]],
    	          [ kernel2[1,0], (kernel2[1,0]+kernel2[1,1]+kernel2[1,2])/2, (kernel2[1,0]-kernel2[1,1]+kernel2[1,2])/2, kernel2[1,2]],
    		      [ kernel2[2,0], (kernel2[2,0]+kernel2[2,1]+kernel2[2,2])/2, (kernel2[2,0]-kernel2[2,1]+kernel2[2,2])/2, kernel2[2,2]],
    		      [ kernel2[3,0], (kernel2[3,0]+kernel2[3,1]+kernel2[3,2])/2, (kernel2[3,0]-kernel2[3,1]+kernel2[3,2])/2, kernel2[3,2]]]

    in (transpose kernel3)

  let transformData (tile: [][]f32): [4][4]f32 =

    --let BT:[][]f32 = [[1.0,0.0,-1.0,0.0],[0.0,1.0,1.0,0.0],[0.0,-1.0,1.0,0.0],[0.0,1.0,0.0,-1.0]]--data

    let tile2 = [[tile[0,0]-tile[2,0],  tile[0,1]-tile[2,1],  tile[0,2]-tile[2,2],  tile[0,3]-tile[2,3]],
    	        [ tile[1,0]+tile[2,0],  tile[1,1]+tile[2,1],  tile[1,2]+tile[2,2],  tile[1,3]+tile[2,3]],
    		    [-tile[1,0]+tile[2,0], -tile[1,1]+tile[2,1], -tile[1,2]+tile[2,2], -tile[1,3]+tile[2,3]],
    		    [ tile[1,0]-tile[3,0],  tile[1,1]-tile[3,1],  tile[1,2]-tile[3,2],  tile[1,3]-tile[3,3]]]

    let tile3 = [[tile2[0,0]-tile2[0,2],  tile2[1,0]-tile2[1,2],  tile2[2,0]-tile2[2,2],  tile2[3,0]-tile2[3,2]],
    	        [ tile2[0,1]+tile2[0,2],  tile2[1,1]+tile2[1,2],  tile2[2,1]+tile2[2,2],  tile2[3,1]+tile2[3,2]],
    		    [-tile2[0,1]+tile2[0,2], -tile2[1,1]+tile2[1,2], -tile2[2,1]+tile2[2,2], -tile2[3,1]+tile2[3,2]],
    		    [ tile2[0,1]-tile2[0,3],  tile2[1,1]-tile2[1,3],  tile2[2,1]-tile2[2,3],  tile2[3,1]-tile2[3,3]]]

    map (\i ->
	  map (\j ->
            unsafe		
	        (winogradConvolution data[i:i+4,j:j+4] t_kernel))
                (range 0 (h_tiles*2) 2))))
          (range 0 (v_tiles*2) 2)

    in (transpose tile3)

  let transformOutput (mult: [][]f32): [4][4]f32 =

    --let AT:[][]f32 = [[1.0,1.0,1.0,0.0],[0.0,1.0,-1.0,-1.0]]--output

    let mult2 = [[mult[0,0]+mult[1,0]+mult[2,0], mult[0,1]+mult[1,1]+mult[2,1], mult[0,2]+mult[1,2]+mult[2,2], mult[0,3]+mult[1,3]+mult[2,3]],
    		    [ mult[1,0]-mult[2,0]-mult[3,0], mult[1,1]-mult[2,1]-mult[3,1], mult[1,2]-mult[2,2]-mult[3,2], mult[1,3]-mult[2,3]-mult[3,3]]]

    let mult3 = [[mult2[0,0]+mult2[0,1]+mult2[0,2], mult2[1,0]+mult2[1,1]+mult2[1,2]],
    		    [ mult2[0,1]-mult2[0,2]-mult2[0,3], mult2[1,1]-mult2[1,2]-mult2[1,3]]]

    in (transpose mult3)

  
  let winogradConvolution [rows][cols]
		                      (tile:  [rows][cols]f32)
		                      (t_kernel: [4][4]f32): [2][2]f32 =

    let BT:[][]f32 = [[1.0,0.0,-1.0,0.0],[0.0,1.0,1.0,0.0],[0.0,-1.0,1.0,0.0],[0.0,1.0,0.0,-1.0]]--data
    let AT:[][]f32 = [[1.0,1.0,1.0,0.0],[0.0,1.0,-1.0,-1.0]]--output
		   
    let t_data = matmul.main (matmul.main BT tile) (transpose BT)
    --let t_data = transformData (tile) 
    let t_mult = pointwise t_data t_kernel 
    
    let res = matmul.main(matmul.main AT t_mult) (transpose AT)
    --let res = transformOutput(t_mult)
    in res
  

  let convolveTilesFirst [rows][cols]
                    (data: [rows][cols]f32) (t_kernel: [4][4]f32)
		                (h_tiles:i32) (v_tiles:i32) : [][]f32 =
    let res =
      map (\i ->
	        flatten (transpose (map (\j ->
                                  unsafe		
	                                (winogradConvolution data[i:i+4,j:j+4] t_kernel))
                (range 0 (h_tiles*2) 2))))
          (range 0 (v_tiles*2) 2)
    in unflatten (rows-2) (cols-2) (flatten (flatten res))


  let convolveTilesSecond [rows][cols]
                    (data: [rows][cols]f32) (t_kernel: [4][4]f32)
                    (h_tiles:i32) (v_tiles:i32) : [][]f32 =
    let res =
      map (\i ->
          flatten (transpose (map (\j ->
                                  unsafe    
                                  (winogradConvolution ([[data[i,j], data[i,j+1], data[i,j+2], data[i,j+3]],
                                                        [data[i+1,j], data[i+1,j+1], data[i+1,j+2], data[i+1,j+3]],
                                                        [data[i+2,j], data[i+2,j+1], data[i+2,j+2], data[i+2,j+3]],
                                                        [data[i+3,j], data[i+3,j+1], data[i+3,j+2], data[i+3,j+3]]]) t_kernel))
                (range 0 (h_tiles*2) 2))))
          (range 0 (v_tiles*2) 2)
    in unflatten (rows-2) (cols-2) (flatten (flatten res))
	
	
  let main [rows_data][cols_data] [rows_kernel][cols_kernel]
           (image: [rows_data][cols_data]f32) (kernel: [rows_kernel][cols_kernel]f32): [][]f32 =

    let G:[][]f32  = [[1.0,0.0,0.0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0.0,0.0,1.0]]--kernel
    let t_kernel = matmul.main (matmul.main G kernel) (transpose G)
    --let t_kernel = transformKernel kernel

    let padded = pad.padImage image
    let horizontal_tiles = cols_data / 2
    let vertical_tiles = rows_data / 2
    let res =
      if (((rows_data*cols_data) % 4) == 0) then
        convolveTilesSecond padded t_kernel horizontal_tiles vertical_tiles
      else
        [[2]] --need an error here
    in res
}
-- ==
-- compiled input @ ../data/pyarray.txt
-- output @ ../data/pyresult.txt
