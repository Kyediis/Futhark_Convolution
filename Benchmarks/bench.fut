-- ==
-- entry: direct im2col winograd mec
-- random input { [2048][2048]f32 [3][3]f32 }
-- random input { [1024][1024]f32 [3][3]f32 }
-- random input { [512][512]f32 [3][3]f32 }

import "../Convolutional_algorithms/direct"
import "../Convolutional_algorithms/im2col"
import "../Convolutional_algorithms/mec"
import "../Convolutional_algorithms/winograd"

entry direct = direct.main
entry im2col = im2col.main
entry mec = mec.main
entry winograd = winograd.main