def EPSILON = 0.01f32

-- multiply a point (p,1) by a 4x4 matrix
def transform_point_4x3 (p: (f32, f32, f32)) (T: [4][4]f32) : (f32, f32, f32) =
  ( T[0][0] * p.0 + T[0][1] * p.1 + T[0][2] * p.2 + T[0][3]
  , T[1][0] * p.0 + T[1][1] * p.1 + T[1][2] * p.2 + T[1][3]
  , T[2][0] * p.0 + T[2][1] * p.1 + T[2][2] * p.2 + T[2][3]
  )

-- multiply a point p by a 3x3 matrix 
def transform_point_3x3 (p: (f32, f32, f32)) (T: [3][3]f32) : (f32, f32, f32) =
  ( T[0][0] * p.0 + T[0][1] * p.1 + T[0][2] * p.2
  , T[1][0] * p.0 + T[1][1] * p.1 + T[1][2] * p.2
  , T[2][0] * p.0 + T[2][1] * p.1 + T[2][2] * p.2
  )

-- convert a coordinate from a pixel value to normalized device coordinates (-1 to 1)
def pix_to_ndc (v: i64) (s: i64) : f32 =
  (2.0 * (f32.i64 v) + 1.0) / (f32.i64 s) - 1.0

-- evaluate the gaussian defined by the given conic and mean at point x
def gaussian (x: (f32, f32, f32)) (coef: f32) (conic: [3][3]f32) (mean: [3]f32) =
  let p = (x.0 - mean[0], x.1 - mean[1], x.2 - mean[2])
  let prod = transform_point_3x3 p conic
  let power = p.0 * prod.0 + p.1 * prod.1 + p.2 * prod.2
  in coef * (f32.exp <| -0.5 * power)

def radiance (p: (i64, i64))            
             (focal_length: f32)         
             (W: i64)                    
             (cam2world: [4][4]f32)      
             (coef: f32)
             (conic: [3][3]f32)
             (delta: f32)
             (max_depth: f32)
             (mean: [3]f32) 
             (ortho_view_box: f32): f32 = --ortho_view_box > 0 for orthographic projection
  -- the pixel in ndc coordinates
  let p_ndc = (pix_to_ndc p.0 W, -1 * pix_to_ndc p.1 W)
  let f = focal_length

  -- distance of this pixel from the center of projection
  let ell = if ortho_view_box > 0   
              then f32.sqrt <| ((p_ndc.0 / f) * (p_ndc.0 / f)) + ((p_ndc.1 / f) * (p_ndc.1 / f)) + 1
              else 1
  let (final_radiance, _, _) =
    loop (radiance, T, depth) = (0f32, 1f32, -ell) -- start the loop with negative depth. use the opengl convention where -Z points "forward"
    while T > EPSILON && depth > max_depth do
      -- enter the loop with ray-space coordinate r = (p, depth). Convert r to camera space
      -- equation 27 in zwicker et al.
      let p_camera = if ortho_view_box > 0
                        then (p_ndc.0 * ortho_view_box, p_ndc.1 * ortho_view_box, depth)
                        else (depth * p_ndc.0 / (f * ell), depth * p_ndc.1 / (f * ell), depth / ell)
      
      -- convert camera-space coordinate to world space
      let p_world = transform_point_4x3 p_camera cam2world 

      -- evaluate the gaussian in world space
      let g_val = gaussian p_world coef conic mean
      let power = f32.exp (-g_val * delta)
      
      -- the contribution of this sample point to the total radiance along the viewing ray
      let radiance_inc = (1 - power) * T

      -- update transmittance
      let T' = T * power

      -- for the next iteration, subtract depth to translate along the viewing ray more "forward" away 
      -- from the center of projection
      in if power < EPSILON then (radiance, T, depth - delta) else (radiance + radiance_inc, T', depth - delta) 
    in final_radiance

entry rasterize (W: i64)                -- WxW is the resolution of the frame. 
                (f: f32)                -- the distance from the center of projection to the projection plane
                (cam2world: [4][4]f32)  -- a matrix that converts camera space points to world space
                (coef: f32)             -- The coefficient in front of the "power" part of the gaussian. Constant for a frame.
                (conic: [3][3]f32)      -- the inverse covariance matrix of the gaussian
                (mean: [3]f32)          -- the mean of the gaussian
                (delta: f32)            -- the euclidean distance on each viewing ray between individual sample points
                (max_depth: f32)        -- the euclidean length of the portion of each viewing ray that will be sampled (starting from the center of projection)
                (ortho_view_box: f32)   -- Does the user want an orthographic projection? If not, leave this 0.
                                  : [W][W]f32 =  
  -- remember to swap x and y in the lambda function because +y means lower on the screen for computers
  tabulate_2d W W (\y x -> radiance (x, y) f W cam2world coef conic delta (-1 * max_depth) mean ortho_view_box)
