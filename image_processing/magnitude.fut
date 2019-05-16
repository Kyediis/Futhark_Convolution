import "futlib/math"

let splitIntoChannels [rows][cols]
                      (image: [rows][cols][3]u8): ([rows][cols]f32,
                                                   [rows][cols]f32,
                                                   [rows][cols]f32) =

  unzip3 (map (\row ->
                unzip3 (map(\pixel ->
                             (f32.u8(pixel[0]) / 255f32,
                              f32.u8(pixel[1]) / 255f32,
                              f32.u8(pixel[2]) / 255f32))
                           row))
              image)

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

let magnitude [rows][cols]
         (imageOne: [rows][cols]f32) (imageTwo: [rows][cols]f32): [rows][cols]f32 =

  unsafe
  map (\row ->
    map (\col ->
          intrinsics.sqrt32((imageOne[row,col]**2 + imageTwo[row,col]**2)))
        (0...cols-1))
      (0...rows-1)

let main [rows][cols]
         (imageOne: [rows][cols][3]u8) (imageTwo: [rows][cols][3]u8): [rows][cols][3]u8 =
  -- First we split the image apart into component channels.
  let (rs1, gs1, bs1) = splitIntoChannels imageOne
  let (rs2, gs2, bs2) = splitIntoChannels imageTwo

  let rs = magnitude rs1 rs2
  let gs = magnitude gs1 gs2
  let bs = magnitude bs1 bs2
  in combineChannels rs gs bs