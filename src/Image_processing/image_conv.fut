import "../Convolutional_algorithms/direct"
-- Split the original three-dimensional array into three
-- two-dimensional arrays of floats: one per colour channel.  The
-- elements of the arrays will have a value from 0 to 1.0.
let splitIntoChannels [rows][cols]
                      (image: [rows][cols][3]u8): ([rows][cols]f32,
                                                   [rows][cols]f32,
                                                   [rows][cols]f32) =
  -- The maps themselves will return an array of triples, so we use
  -- unzip to turn it into a triple of arrays.  Due to the way the
  -- Futhark compiler represents arrays in the generated code, zip and
  -- unzip are entirely free.
  unzip3 (map (\row ->
                unzip3 (map(\pixel ->
                             (f32.u8(pixel[0]) / 255f32,
                              f32.u8(pixel[1]) / 255f32,
                              f32.u8(pixel[2]) / 255f32))
                           row))
              image)

-- The inverse of splitIntoChannels.
let combineChannels [rows][cols]
                    (rs: [rows][cols]f32)
                    (gs: [rows][cols]f32)
                    (bs: [rows][cols]f32): [rows][cols][3]u8 =
  map3 (\rs_row gs_row bs_row ->
         map3 (\r g b ->
                [u8.f32(r * 255f32),
                 u8.f32(g * 255f32),
                 u8.f32(b * 255f32)])
             rs_row gs_row bs_row)
      rs gs bs

-- Perform the specified number of blurring operations on the image.
let main [rows][cols]
         (iterations: i32) (image: [rows][cols][3]u8) (kernel: [3][3]f32): [rows][cols][3]u8 =
  -- First we split the image apart into component channels.
  let (rs, gs, bs) = splitIntoChannels image
  -- Then we loop 'iterations' times.
  let (rs, gs, bs) = loop (rs, gs, bs) for _i < iterations do
    -- Blur each channel by itself.  The Futhark compiler will fuse
    -- these together into just one loop.
    let rs = direct.main rs kernel
    let gs = direct.main gs kernel
    let bs = direct.main bs kernel
    in (rs, gs, bs)
  -- Finally, combine the separate channels back into a single image.
  in combineChannels rs gs bs