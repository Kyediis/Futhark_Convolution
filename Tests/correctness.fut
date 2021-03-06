import "../src/Convolutional_algorithms/direct"
import "../src/Convolutional_algorithms/im2col"
import "../src/Convolutional_algorithms/mec"
import "../src/Convolutional_algorithms/winograd"

-- ==
-- entry: direct_test
-- compiled input @ pyarray.in
-- output @ pyresult.out

entry direct_test = direct.main

-- ==
-- entry: im2col_test
-- compiled input @ pyarray.in
-- output @ pyresult.out

entry im2col_test = im2col.main

-- ==
-- entry: mec_test
-- compiled input @ pyarray.in
-- output @ pyresult.out

entry mec_test = mec.main

-- ==
-- entry: winograd_test
-- compiled input @ pyarray.in
-- output @ pyresult.out

entry winograd_test = winograd.main
