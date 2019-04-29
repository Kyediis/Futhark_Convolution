import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, value) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and self.device.type == device_type:
               if type(value) == str:
                   sizes[size] = self.device.get_info(getattr(cl.device_info,value))
               else:
                   sizes[size] = value
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0
    self.free_list = {}

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            raise Exception('Unknown size class for size \'{}\': {}'.format(k, v['class']))
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
__kernel void map_13463(int32_t sizze_13091, int32_t sizze_13092,
                        int32_t sizze_13093, __global
                        unsigned char *image_mem_13964, __global
                        unsigned char *mem_13970, __global
                        unsigned char *mem_13974, __global
                        unsigned char *mem_13978)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14115;
    int32_t group_sizze_14116;
    int32_t gtid_13454;
    int32_t gtid_13455;
    int32_t global_tid_13463;
    int32_t local_tid_13464;
    int32_t group_id_13465;
    
    global_tid_13463 = get_global_id(0);
    local_tid_13464 = get_local_id(0);
    group_sizze_14116 = get_local_size(0);
    wave_sizze_14115 = LOCKSTEP_WIDTH;
    group_id_13465 = get_group_id(0);
    gtid_13454 = squot32(global_tid_13463, sizze_13092);
    gtid_13455 = global_tid_13463 - squot32(global_tid_13463, sizze_13092) *
        sizze_13092;
    
    int8_t arg_13467;
    float res_13468;
    float res_13469;
    int8_t arg_13470;
    float res_13471;
    float res_13472;
    int8_t arg_13473;
    float res_13474;
    float res_13475;
    
    if (slt32(gtid_13454, sizze_13091) && slt32(gtid_13455, sizze_13092)) {
        arg_13467 = *(__global int8_t *) &image_mem_13964[gtid_13454 *
                                                          (sizze_13093 *
                                                           sizze_13092) +
                                                          gtid_13455 *
                                                          sizze_13093];
        res_13468 = uitofp_i8_f32(arg_13467);
        res_13469 = res_13468 / 255.0F;
        arg_13470 = *(__global int8_t *) &image_mem_13964[gtid_13454 *
                                                          (sizze_13093 *
                                                           sizze_13092) +
                                                          gtid_13455 *
                                                          sizze_13093 + 1];
        res_13471 = uitofp_i8_f32(arg_13470);
        res_13472 = res_13471 / 255.0F;
        arg_13473 = *(__global int8_t *) &image_mem_13964[gtid_13454 *
                                                          (sizze_13093 *
                                                           sizze_13092) +
                                                          gtid_13455 *
                                                          sizze_13093 + 2];
        res_13474 = uitofp_i8_f32(arg_13473);
        res_13475 = res_13474 / 255.0F;
    }
    if (slt32(gtid_13454, sizze_13091) && slt32(gtid_13455, sizze_13092)) {
        *(__global float *) &mem_13970[(gtid_13454 * sizze_13092 + gtid_13455) *
                                       4] = res_13469;
    }
    if (slt32(gtid_13454, sizze_13091) && slt32(gtid_13455, sizze_13092)) {
        *(__global float *) &mem_13974[(gtid_13454 * sizze_13092 + gtid_13455) *
                                       4] = res_13472;
    }
    if (slt32(gtid_13454, sizze_13091) && slt32(gtid_13455, sizze_13092)) {
        *(__global float *) &mem_13978[(gtid_13454 * sizze_13092 + gtid_13455) *
                                       4] = res_13475;
    }
}
__kernel void map_13499(int32_t sizze_13142, int32_t range_end_13151,
                        int32_t num_elems_13154, int32_t range_end_13156,
                        int32_t num_elems_13159, __global
                        unsigned char *rs_mem_13980, __global
                        unsigned char *mem_13988)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14132;
    int32_t group_sizze_14133;
    int32_t gtid_13490;
    int32_t gtid_13491;
    int32_t global_tid_13499;
    int32_t local_tid_13500;
    int32_t group_id_13501;
    
    global_tid_13499 = get_global_id(0);
    local_tid_13500 = get_local_id(0);
    group_sizze_14133 = get_local_size(0);
    wave_sizze_14132 = LOCKSTEP_WIDTH;
    group_id_13501 = get_group_id(0);
    gtid_13490 = squot32(global_tid_13499, num_elems_13159);
    gtid_13491 = global_tid_13499 - squot32(global_tid_13499, num_elems_13159) *
        num_elems_13159;
    
    bool binop_x_13903;
    bool binop_y_13904;
    bool index_primexp_13905;
    bool res_13505;
    bool x_13506;
    bool res_13507;
    bool x_13508;
    float res_13509;
    
    if (slt32(gtid_13490, num_elems_13154) && slt32(gtid_13491,
                                                    num_elems_13159)) {
        binop_x_13903 = slt32(0, gtid_13490);
        binop_y_13904 = slt32(gtid_13490, range_end_13151);
        index_primexp_13905 = binop_x_13903 && binop_y_13904;
        res_13505 = slt32(0, gtid_13491);
        x_13506 = res_13505 && index_primexp_13905;
        res_13507 = slt32(gtid_13491, range_end_13156);
        x_13508 = x_13506 && res_13507;
        if (x_13508) {
            int32_t i_13510;
            int32_t i_13511;
            float res_13512;
            
            i_13510 = gtid_13490 - 1;
            i_13511 = gtid_13491 - 1;
            res_13512 = *(__global float *) &rs_mem_13980[(i_13510 *
                                                           sizze_13142 +
                                                           i_13511) * 4];
            res_13509 = res_13512;
        } else {
            res_13509 = 0.0F;
        }
    }
    if (slt32(gtid_13490, num_elems_13154) && slt32(gtid_13491,
                                                    num_elems_13159)) {
        *(__global float *) &mem_13988[(gtid_13490 * num_elems_13159 +
                                        gtid_13491) * 4] = res_13509;
    }
}
__kernel void map_13535(int32_t num_elems_13159, int32_t num_elems_13194,
                        int32_t flat_dim_13196, __global
                        unsigned char *mem_13988, __global
                        unsigned char *mem_13992)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14134;
    int32_t group_sizze_14135;
    int32_t gtid_13528;
    int32_t global_tid_13535;
    int32_t local_tid_13536;
    int32_t group_id_13537;
    
    global_tid_13535 = get_global_id(0);
    local_tid_13536 = get_local_id(0);
    group_sizze_14135 = get_local_size(0);
    wave_sizze_14134 = LOCKSTEP_WIDTH;
    group_id_13537 = get_group_id(0);
    gtid_13528 = global_tid_13535;
    
    int32_t index_primexp_13883;
    int32_t j_13540;
    int32_t j_m_i_13541;
    
    if (slt32(gtid_13528, num_elems_13194)) {
        index_primexp_13883 = 1 + gtid_13528;
        j_13540 = 2 + index_primexp_13883;
        j_m_i_13541 = j_13540 - gtid_13528;
    }
    if (slt32(gtid_13528, num_elems_13194)) {
        for (int32_t i_14136 = 0; i_14136 < flat_dim_13196; i_14136++) {
            *(__global float *) &mem_13992[(gtid_13528 + i_14136 *
                                            num_elems_13194) * 4] = *(__global
                                                                      float *) &mem_13988[(gtid_13528 +
                                                                                           (squot32(i_14136,
                                                                                                    j_m_i_13541) *
                                                                                            num_elems_13159 +
                                                                                            (i_14136 -
                                                                                             squot32(i_14136,
                                                                                                     j_m_i_13541) *
                                                                                             j_m_i_13541))) *
                                                                                          4];
        }
    }
}
__kernel void map_13551(int32_t num_elems_13194, int32_t flat_dim_13196,
                        int32_t res_13206, __global unsigned char *mem_13996,
                        __global unsigned char *mem_14001)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14137;
    int32_t group_sizze_14138;
    int32_t gtid_13544;
    int32_t global_tid_13551;
    int32_t local_tid_13552;
    int32_t group_id_13553;
    
    global_tid_13551 = get_global_id(0);
    local_tid_13552 = get_local_id(0);
    group_sizze_14138 = get_local_size(0);
    wave_sizze_14137 = LOCKSTEP_WIDTH;
    group_id_13553 = get_group_id(0);
    gtid_13544 = global_tid_13551;
    
    int32_t res_13555;
    
    if (slt32(gtid_13544, res_13206)) {
        res_13555 = 3 * gtid_13544;
    }
    if (slt32(gtid_13544, res_13206)) {
        for (int32_t i_14139 = 0; i_14139 < num_elems_13194; i_14139++) {
            for (int32_t i_14140 = 0; i_14140 < 9; i_14140++) {
                *(__global float *) &mem_14001[(res_13206 * 9 * 0 + res_13206 *
                                                0 + gtid_13544 + (i_14139 *
                                                                  (res_13206 *
                                                                   9) +
                                                                  i_14140 *
                                                                  res_13206)) *
                                               4] = *(__global
                                                      float *) &mem_13996[(res_13555 +
                                                                           (i_14139 *
                                                                            flat_dim_13196 +
                                                                            i_14140)) *
                                                                          4];
            }
        }
    }
}
__kernel void map_13577(int32_t sizze_13095, int32_t num_elems_13194,
                        int32_t res_13206, int32_t num_elems_13215,
                        int32_t num_elems_13220, __global
                        unsigned char *kernel_mem_13966, __global
                        unsigned char *mem_14011, __global
                        unsigned char *mem_14015)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14141;
    int32_t group_sizze_14142;
    int32_t gtid_13568;
    int32_t gtid_13569;
    int32_t global_tid_13577;
    int32_t local_tid_13578;
    int32_t group_id_13579;
    
    global_tid_13577 = get_global_id(0);
    local_tid_13578 = get_local_id(0);
    group_sizze_14142 = get_local_size(0);
    wave_sizze_14141 = LOCKSTEP_WIDTH;
    group_id_13579 = get_group_id(0);
    gtid_13568 = squot32(global_tid_13577, num_elems_13220);
    gtid_13569 = global_tid_13577 - squot32(global_tid_13577, num_elems_13220) *
        num_elems_13220;
    
    float res_13583;
    
    if (slt32(gtid_13568, num_elems_13215) && slt32(gtid_13569,
                                                    num_elems_13220)) {
        float x_13586 = 0.0F;
        
        for (int32_t chunk_offset_13585 = 0; chunk_offset_13585 < 9;
             chunk_offset_13585++) {
            float x_13595;
            int32_t new_index_13931;
            int32_t binop_y_13933;
            int32_t new_index_13934;
            float x_13596;
            float res_13598;
            float res_13600;
            
            x_13595 = *(__global float *) &mem_14011[(chunk_offset_13585 *
                                                      (num_elems_13194 *
                                                       res_13206) + gtid_13568 *
                                                      num_elems_13194 +
                                                      gtid_13569) * 4];
            new_index_13931 = squot32(chunk_offset_13585, sizze_13095);
            binop_y_13933 = sizze_13095 * new_index_13931;
            new_index_13934 = chunk_offset_13585 - binop_y_13933;
            x_13596 = *(__global float *) &kernel_mem_13966[(new_index_13931 *
                                                             sizze_13095 +
                                                             new_index_13934) *
                                                            4];
            res_13598 = x_13595 * x_13596;
            res_13600 = x_13586 + res_13598;
            
            float x_tmp_14143 = res_13600;
            
            x_13586 = x_tmp_14143;
        }
        res_13583 = x_13586;
    }
    if (slt32(gtid_13568, num_elems_13215) && slt32(gtid_13569,
                                                    num_elems_13220)) {
        *(__global float *) &mem_14015[(gtid_13568 * num_elems_13220 +
                                        gtid_13569) * 4] = res_13583;
    }
}
__kernel void map_13624(int32_t sizze_13144, int32_t range_end_13234,
                        int32_t num_elems_13237, int32_t range_end_13239,
                        int32_t num_elems_13242, __global
                        unsigned char *gs_mem_13982, __global
                        unsigned char *mem_14019)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14144;
    int32_t group_sizze_14145;
    int32_t gtid_13615;
    int32_t gtid_13616;
    int32_t global_tid_13624;
    int32_t local_tid_13625;
    int32_t group_id_13626;
    
    global_tid_13624 = get_global_id(0);
    local_tid_13625 = get_local_id(0);
    group_sizze_14145 = get_local_size(0);
    wave_sizze_14144 = LOCKSTEP_WIDTH;
    group_id_13626 = get_group_id(0);
    gtid_13615 = squot32(global_tid_13624, num_elems_13242);
    gtid_13616 = global_tid_13624 - squot32(global_tid_13624, num_elems_13242) *
        num_elems_13242;
    
    bool binop_x_13906;
    bool binop_y_13907;
    bool index_primexp_13908;
    bool res_13630;
    bool x_13631;
    bool res_13632;
    bool x_13633;
    float res_13634;
    
    if (slt32(gtid_13615, num_elems_13237) && slt32(gtid_13616,
                                                    num_elems_13242)) {
        binop_x_13906 = slt32(0, gtid_13615);
        binop_y_13907 = slt32(gtid_13615, range_end_13234);
        index_primexp_13908 = binop_x_13906 && binop_y_13907;
        res_13630 = slt32(0, gtid_13616);
        x_13631 = res_13630 && index_primexp_13908;
        res_13632 = slt32(gtid_13616, range_end_13239);
        x_13633 = x_13631 && res_13632;
        if (x_13633) {
            int32_t i_13635;
            int32_t i_13636;
            float res_13637;
            
            i_13635 = gtid_13615 - 1;
            i_13636 = gtid_13616 - 1;
            res_13637 = *(__global float *) &gs_mem_13982[(i_13635 *
                                                           sizze_13144 +
                                                           i_13636) * 4];
            res_13634 = res_13637;
        } else {
            res_13634 = 0.0F;
        }
    }
    if (slt32(gtid_13615, num_elems_13237) && slt32(gtid_13616,
                                                    num_elems_13242)) {
        *(__global float *) &mem_14019[(gtid_13615 * num_elems_13242 +
                                        gtid_13616) * 4] = res_13634;
    }
}
__kernel void map_13660(int32_t num_elems_13242, int32_t num_elems_13277,
                        int32_t flat_dim_13279, __global
                        unsigned char *mem_14019, __global
                        unsigned char *mem_14023)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14146;
    int32_t group_sizze_14147;
    int32_t gtid_13653;
    int32_t global_tid_13660;
    int32_t local_tid_13661;
    int32_t group_id_13662;
    
    global_tid_13660 = get_global_id(0);
    local_tid_13661 = get_local_id(0);
    group_sizze_14147 = get_local_size(0);
    wave_sizze_14146 = LOCKSTEP_WIDTH;
    group_id_13662 = get_group_id(0);
    gtid_13653 = global_tid_13660;
    
    int32_t index_primexp_13891;
    int32_t j_13665;
    int32_t j_m_i_13666;
    
    if (slt32(gtid_13653, num_elems_13277)) {
        index_primexp_13891 = 1 + gtid_13653;
        j_13665 = 2 + index_primexp_13891;
        j_m_i_13666 = j_13665 - gtid_13653;
    }
    if (slt32(gtid_13653, num_elems_13277)) {
        for (int32_t i_14148 = 0; i_14148 < flat_dim_13279; i_14148++) {
            *(__global float *) &mem_14023[(gtid_13653 + i_14148 *
                                            num_elems_13277) * 4] = *(__global
                                                                      float *) &mem_14019[(gtid_13653 +
                                                                                           (squot32(i_14148,
                                                                                                    j_m_i_13666) *
                                                                                            num_elems_13242 +
                                                                                            (i_14148 -
                                                                                             squot32(i_14148,
                                                                                                     j_m_i_13666) *
                                                                                             j_m_i_13666))) *
                                                                                          4];
        }
    }
}
__kernel void map_13676(int32_t num_elems_13277, int32_t flat_dim_13279,
                        int32_t res_13289, __global unsigned char *mem_14027,
                        __global unsigned char *mem_14032)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14149;
    int32_t group_sizze_14150;
    int32_t gtid_13669;
    int32_t global_tid_13676;
    int32_t local_tid_13677;
    int32_t group_id_13678;
    
    global_tid_13676 = get_global_id(0);
    local_tid_13677 = get_local_id(0);
    group_sizze_14150 = get_local_size(0);
    wave_sizze_14149 = LOCKSTEP_WIDTH;
    group_id_13678 = get_group_id(0);
    gtid_13669 = global_tid_13676;
    
    int32_t res_13680;
    
    if (slt32(gtid_13669, res_13289)) {
        res_13680 = 3 * gtid_13669;
    }
    if (slt32(gtid_13669, res_13289)) {
        for (int32_t i_14151 = 0; i_14151 < num_elems_13277; i_14151++) {
            for (int32_t i_14152 = 0; i_14152 < 9; i_14152++) {
                *(__global float *) &mem_14032[(res_13289 * 9 * 0 + res_13289 *
                                                0 + gtid_13669 + (i_14151 *
                                                                  (res_13289 *
                                                                   9) +
                                                                  i_14152 *
                                                                  res_13289)) *
                                               4] = *(__global
                                                      float *) &mem_14027[(res_13680 +
                                                                           (i_14151 *
                                                                            flat_dim_13279 +
                                                                            i_14152)) *
                                                                          4];
            }
        }
    }
}
__kernel void map_13702(int32_t sizze_13095, int32_t num_elems_13277,
                        int32_t res_13289, int32_t num_elems_13298,
                        int32_t num_elems_13303, __global
                        unsigned char *kernel_mem_13966, __global
                        unsigned char *mem_14042, __global
                        unsigned char *mem_14046)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14153;
    int32_t group_sizze_14154;
    int32_t gtid_13693;
    int32_t gtid_13694;
    int32_t global_tid_13702;
    int32_t local_tid_13703;
    int32_t group_id_13704;
    
    global_tid_13702 = get_global_id(0);
    local_tid_13703 = get_local_id(0);
    group_sizze_14154 = get_local_size(0);
    wave_sizze_14153 = LOCKSTEP_WIDTH;
    group_id_13704 = get_group_id(0);
    gtid_13693 = squot32(global_tid_13702, num_elems_13303);
    gtid_13694 = global_tid_13702 - squot32(global_tid_13702, num_elems_13303) *
        num_elems_13303;
    
    float res_13708;
    
    if (slt32(gtid_13693, num_elems_13298) && slt32(gtid_13694,
                                                    num_elems_13303)) {
        float x_13711 = 0.0F;
        
        for (int32_t chunk_offset_13710 = 0; chunk_offset_13710 < 9;
             chunk_offset_13710++) {
            float x_13720;
            int32_t new_index_13945;
            int32_t binop_y_13947;
            int32_t new_index_13948;
            float x_13721;
            float res_13723;
            float res_13725;
            
            x_13720 = *(__global float *) &mem_14042[(chunk_offset_13710 *
                                                      (num_elems_13277 *
                                                       res_13289) + gtid_13693 *
                                                      num_elems_13277 +
                                                      gtid_13694) * 4];
            new_index_13945 = squot32(chunk_offset_13710, sizze_13095);
            binop_y_13947 = sizze_13095 * new_index_13945;
            new_index_13948 = chunk_offset_13710 - binop_y_13947;
            x_13721 = *(__global float *) &kernel_mem_13966[(new_index_13945 *
                                                             sizze_13095 +
                                                             new_index_13948) *
                                                            4];
            res_13723 = x_13720 * x_13721;
            res_13725 = x_13711 + res_13723;
            
            float x_tmp_14155 = res_13725;
            
            x_13711 = x_tmp_14155;
        }
        res_13708 = x_13711;
    }
    if (slt32(gtid_13693, num_elems_13298) && slt32(gtid_13694,
                                                    num_elems_13303)) {
        *(__global float *) &mem_14046[(gtid_13693 * num_elems_13303 +
                                        gtid_13694) * 4] = res_13708;
    }
}
__kernel void map_13749(int32_t sizze_13146, int32_t range_end_13317,
                        int32_t num_elems_13320, int32_t range_end_13322,
                        int32_t num_elems_13325, __global
                        unsigned char *bs_mem_13984, __global
                        unsigned char *mem_14050)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14156;
    int32_t group_sizze_14157;
    int32_t gtid_13740;
    int32_t gtid_13741;
    int32_t global_tid_13749;
    int32_t local_tid_13750;
    int32_t group_id_13751;
    
    global_tid_13749 = get_global_id(0);
    local_tid_13750 = get_local_id(0);
    group_sizze_14157 = get_local_size(0);
    wave_sizze_14156 = LOCKSTEP_WIDTH;
    group_id_13751 = get_group_id(0);
    gtid_13740 = squot32(global_tid_13749, num_elems_13325);
    gtid_13741 = global_tid_13749 - squot32(global_tid_13749, num_elems_13325) *
        num_elems_13325;
    
    bool binop_x_13909;
    bool binop_y_13910;
    bool index_primexp_13911;
    bool res_13755;
    bool x_13756;
    bool res_13757;
    bool x_13758;
    float res_13759;
    
    if (slt32(gtid_13740, num_elems_13320) && slt32(gtid_13741,
                                                    num_elems_13325)) {
        binop_x_13909 = slt32(0, gtid_13740);
        binop_y_13910 = slt32(gtid_13740, range_end_13317);
        index_primexp_13911 = binop_x_13909 && binop_y_13910;
        res_13755 = slt32(0, gtid_13741);
        x_13756 = res_13755 && index_primexp_13911;
        res_13757 = slt32(gtid_13741, range_end_13322);
        x_13758 = x_13756 && res_13757;
        if (x_13758) {
            int32_t i_13760;
            int32_t i_13761;
            float res_13762;
            
            i_13760 = gtid_13740 - 1;
            i_13761 = gtid_13741 - 1;
            res_13762 = *(__global float *) &bs_mem_13984[(i_13760 *
                                                           sizze_13146 +
                                                           i_13761) * 4];
            res_13759 = res_13762;
        } else {
            res_13759 = 0.0F;
        }
    }
    if (slt32(gtid_13740, num_elems_13320) && slt32(gtid_13741,
                                                    num_elems_13325)) {
        *(__global float *) &mem_14050[(gtid_13740 * num_elems_13325 +
                                        gtid_13741) * 4] = res_13759;
    }
}
__kernel void map_13785(int32_t num_elems_13325, int32_t num_elems_13360,
                        int32_t flat_dim_13362, __global
                        unsigned char *mem_14050, __global
                        unsigned char *mem_14054)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14158;
    int32_t group_sizze_14159;
    int32_t gtid_13778;
    int32_t global_tid_13785;
    int32_t local_tid_13786;
    int32_t group_id_13787;
    
    global_tid_13785 = get_global_id(0);
    local_tid_13786 = get_local_id(0);
    group_sizze_14159 = get_local_size(0);
    wave_sizze_14158 = LOCKSTEP_WIDTH;
    group_id_13787 = get_group_id(0);
    gtid_13778 = global_tid_13785;
    
    int32_t index_primexp_13899;
    int32_t j_13790;
    int32_t j_m_i_13791;
    
    if (slt32(gtid_13778, num_elems_13360)) {
        index_primexp_13899 = 1 + gtid_13778;
        j_13790 = 2 + index_primexp_13899;
        j_m_i_13791 = j_13790 - gtid_13778;
    }
    if (slt32(gtid_13778, num_elems_13360)) {
        for (int32_t i_14160 = 0; i_14160 < flat_dim_13362; i_14160++) {
            *(__global float *) &mem_14054[(gtid_13778 + i_14160 *
                                            num_elems_13360) * 4] = *(__global
                                                                      float *) &mem_14050[(gtid_13778 +
                                                                                           (squot32(i_14160,
                                                                                                    j_m_i_13791) *
                                                                                            num_elems_13325 +
                                                                                            (i_14160 -
                                                                                             squot32(i_14160,
                                                                                                     j_m_i_13791) *
                                                                                             j_m_i_13791))) *
                                                                                          4];
        }
    }
}
__kernel void map_13801(int32_t num_elems_13360, int32_t flat_dim_13362,
                        int32_t res_13372, __global unsigned char *mem_14058,
                        __global unsigned char *mem_14063)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14161;
    int32_t group_sizze_14162;
    int32_t gtid_13794;
    int32_t global_tid_13801;
    int32_t local_tid_13802;
    int32_t group_id_13803;
    
    global_tid_13801 = get_global_id(0);
    local_tid_13802 = get_local_id(0);
    group_sizze_14162 = get_local_size(0);
    wave_sizze_14161 = LOCKSTEP_WIDTH;
    group_id_13803 = get_group_id(0);
    gtid_13794 = global_tid_13801;
    
    int32_t res_13805;
    
    if (slt32(gtid_13794, res_13372)) {
        res_13805 = 3 * gtid_13794;
    }
    if (slt32(gtid_13794, res_13372)) {
        for (int32_t i_14163 = 0; i_14163 < num_elems_13360; i_14163++) {
            for (int32_t i_14164 = 0; i_14164 < 9; i_14164++) {
                *(__global float *) &mem_14063[(res_13372 * 9 * 0 + res_13372 *
                                                0 + gtid_13794 + (i_14163 *
                                                                  (res_13372 *
                                                                   9) +
                                                                  i_14164 *
                                                                  res_13372)) *
                                               4] = *(__global
                                                      float *) &mem_14058[(res_13805 +
                                                                           (i_14163 *
                                                                            flat_dim_13362 +
                                                                            i_14164)) *
                                                                          4];
            }
        }
    }
}
__kernel void map_13827(int32_t sizze_13095, int32_t num_elems_13360,
                        int32_t res_13372, int32_t num_elems_13381,
                        int32_t num_elems_13386, __global
                        unsigned char *kernel_mem_13966, __global
                        unsigned char *mem_14073, __global
                        unsigned char *mem_14077)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14165;
    int32_t group_sizze_14166;
    int32_t gtid_13818;
    int32_t gtid_13819;
    int32_t global_tid_13827;
    int32_t local_tid_13828;
    int32_t group_id_13829;
    
    global_tid_13827 = get_global_id(0);
    local_tid_13828 = get_local_id(0);
    group_sizze_14166 = get_local_size(0);
    wave_sizze_14165 = LOCKSTEP_WIDTH;
    group_id_13829 = get_group_id(0);
    gtid_13818 = squot32(global_tid_13827, num_elems_13386);
    gtid_13819 = global_tid_13827 - squot32(global_tid_13827, num_elems_13386) *
        num_elems_13386;
    
    float res_13833;
    
    if (slt32(gtid_13818, num_elems_13381) && slt32(gtid_13819,
                                                    num_elems_13386)) {
        float x_13836 = 0.0F;
        
        for (int32_t chunk_offset_13835 = 0; chunk_offset_13835 < 9;
             chunk_offset_13835++) {
            float x_13845;
            int32_t new_index_13959;
            int32_t binop_y_13961;
            int32_t new_index_13962;
            float x_13846;
            float res_13848;
            float res_13850;
            
            x_13845 = *(__global float *) &mem_14073[(chunk_offset_13835 *
                                                      (num_elems_13360 *
                                                       res_13372) + gtid_13818 *
                                                      num_elems_13360 +
                                                      gtid_13819) * 4];
            new_index_13959 = squot32(chunk_offset_13835, sizze_13095);
            binop_y_13961 = sizze_13095 * new_index_13959;
            new_index_13962 = chunk_offset_13835 - binop_y_13961;
            x_13846 = *(__global float *) &kernel_mem_13966[(new_index_13959 *
                                                             sizze_13095 +
                                                             new_index_13962) *
                                                            4];
            res_13848 = x_13845 * x_13846;
            res_13850 = x_13836 + res_13848;
            
            float x_tmp_14167 = res_13850;
            
            x_13836 = x_tmp_14167;
        }
        res_13833 = x_13836;
    }
    if (slt32(gtid_13818, num_elems_13381) && slt32(gtid_13819,
                                                    num_elems_13386)) {
        *(__global float *) &mem_14077[(gtid_13818 * num_elems_13386 +
                                        gtid_13819) * 4] = res_13833;
    }
}
__kernel void map_13860(int32_t sizze_13132, int32_t sizze_13133,
                        int32_t sizze_13135, int32_t sizze_13137,
                        int32_t sizze_13401, __global
                        unsigned char *res_mem_14079, __global
                        unsigned char *res_mem_14081, __global
                        unsigned char *res_mem_14083, __global
                        unsigned char *mem_14085, __global
                        unsigned char *mem_14089)
{
    const int32_t group_sizze_13855 = mainzigroup_sizze_13854;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t wave_sizze_14168;
    int32_t group_sizze_14169;
    int32_t gtid_13851;
    int32_t gtid_13852;
    int32_t global_tid_13860;
    int32_t local_tid_13861;
    int32_t group_id_13862;
    
    global_tid_13860 = get_global_id(0);
    local_tid_13861 = get_local_id(0);
    group_sizze_14169 = get_local_size(0);
    wave_sizze_14168 = LOCKSTEP_WIDTH;
    group_id_13862 = get_group_id(0);
    gtid_13851 = squot32(global_tid_13860, sizze_13401);
    gtid_13852 = global_tid_13860 - squot32(global_tid_13860, sizze_13401) *
        sizze_13401;
    
    float x_13863;
    float x_13864;
    float x_13865;
    float arg_13866;
    int8_t arg_13867;
    float arg_13868;
    int8_t arg_13869;
    float arg_13870;
    int8_t arg_13871;
    
    if (slt32(gtid_13851, sizze_13132) && slt32(gtid_13852, sizze_13401)) {
        x_13863 = *(__global float *) &res_mem_14079[(gtid_13851 * sizze_13133 +
                                                      gtid_13852) * 4];
        x_13864 = *(__global float *) &res_mem_14081[(gtid_13851 * sizze_13135 +
                                                      gtid_13852) * 4];
        x_13865 = *(__global float *) &res_mem_14083[(gtid_13851 * sizze_13137 +
                                                      gtid_13852) * 4];
        arg_13866 = 255.0F * x_13863;
        arg_13867 = fptoui_f32_i8(arg_13866);
        arg_13868 = 255.0F * x_13864;
        arg_13869 = fptoui_f32_i8(arg_13868);
        arg_13870 = 255.0F * x_13865;
        arg_13871 = fptoui_f32_i8(arg_13870);
        *(__global int8_t *) &mem_14085[group_id_13862 * (group_sizze_13855 *
                                                          3) + local_tid_13861 +
                                        0 * group_sizze_13855] = arg_13867;
        *(__global int8_t *) &mem_14085[group_id_13862 * (group_sizze_13855 *
                                                          3) + local_tid_13861 +
                                        group_sizze_13855] = arg_13869;
        *(__global int8_t *) &mem_14085[group_id_13862 * (group_sizze_13855 *
                                                          3) + local_tid_13861 +
                                        2 * group_sizze_13855] = arg_13871;
    }
    if (slt32(gtid_13851, sizze_13132) && slt32(gtid_13852, sizze_13401)) {
        for (int32_t i_14170 = 0; i_14170 < 3; i_14170++) {
            *(__global int8_t *) &mem_14089[gtid_13851 * (3 * sizze_13401) +
                                            gtid_13852 * 3 + i_14170] =
                *(__global int8_t *) &mem_14085[group_id_13862 *
                                                (group_sizze_13855 * 3) +
                                                local_tid_13861 + i_14170 *
                                                group_sizze_13855];
        }
    }
}
__kernel void map_transpose_f32(int32_t destoffset_1, int32_t srcoffset_3,
                                int32_t num_arrays_4, int32_t x_elems_5,
                                int32_t y_elems_6, int32_t in_elems_7,
                                int32_t out_elems_8, int32_t mulx_9,
                                int32_t muly_10, __global
                                unsigned char *destmem_0, __global
                                unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11, 4224);
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,
                                                                 in_elems_7)) {
                *(__local float *) &block_11[((get_local_id_1_39 + j_43 * 8) *
                                              33 + get_local_id_0_38) *
                                             sizeof(float)] = *(__global
                                                                float *) &srcmem_2[(idata_offset_34 +
                                                                                    index_in_35) *
                                                                                   sizeof(float)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,
                                                                 out_elems_8)) {
                *(__global float *) &destmem_0[(odata_offset_33 +
                                                index_out_36) * sizeof(float)] =
                    *(__local float *) &block_11[(get_local_id_0_38 * 33 +
                                                  get_local_id_1_39 + j_43 *
                                                  8) * sizeof(float)];
            }
        }
    }
}
__kernel void map_transpose_f32_low_height(int32_t destoffset_1,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8, int32_t mulx_9,
                                           int32_t muly_10, __global
                                           unsigned char *destmem_0, __global
                                           unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11, 1088);
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_9) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_9);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        *(__local float *) &block_11[(get_local_id_1_39 * 17 +
                                      get_local_id_0_38) * sizeof(float)] =
            *(__global float *) &srcmem_2[(idata_offset_34 + index_in_35) *
                                          sizeof(float)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);
    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_9) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *
                                       sizeof(float)] = *(__local
                                                          float *) &block_11[(get_local_id_0_38 *
                                                                              17 +
                                                                              get_local_id_1_39) *
                                                                             sizeof(float)];
    }
}
__kernel void map_transpose_f32_low_width(int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t in_elems_7,
                                          int32_t out_elems_8, int32_t mulx_9,
                                          int32_t muly_10, __global
                                          unsigned char *destmem_0, __global
                                          unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11, 1088);
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_10);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_10) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        *(__local float *) &block_11[(get_local_id_1_39 * 17 +
                                      get_local_id_0_38) * sizeof(float)] =
            *(__global float *) &srcmem_2[(idata_offset_34 + index_in_35) *
                                          sizeof(float)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_10 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_10) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *
                                       sizeof(float)] = *(__local
                                                          float *) &block_11[(get_local_id_0_38 *
                                                                              17 +
                                                                              get_local_id_1_39) *
                                                                             sizeof(float)];
    }
}
__kernel void map_transpose_f32_small(int32_t destoffset_1, int32_t srcoffset_3,
                                      int32_t num_arrays_4, int32_t x_elems_5,
                                      int32_t y_elems_6, int32_t in_elems_7,
                                      int32_t out_elems_8, int32_t mulx_9,
                                      int32_t muly_10, __global
                                      unsigned char *destmem_0, __global
                                      unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    ALIGNED_LOCAL_MEMORY(block_11, 1);
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, in_elems_7)) {
        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *
                                       sizeof(float)] = *(__global
                                                          float *) &srcmem_2[(idata_offset_34 +
                                                                              index_in_35) *
                                                                             sizeof(float)];
    }
}
"""
# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        map(f.unget_char, read[::-1])
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in string.hexdigits:
            s += c
            c = f.get_char()
        elif c == '_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16))


def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in [b'x', b'X']:
        c = f.get_char() # skip X
        s += parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == '_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
    if len(s) == 0:
        raise ValueError
    return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      s = c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      s = parse_int(f)

    return s

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    for i in range(rank):
        parse_specific_string(f, '[]')
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return None

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank-1)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if elems == None:
        # Empty array
        return np.empty([0]*rank, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype='<'+bin_fmt)
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def write_value(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[]' for _ in v.shape[1:]]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

################################################################################
### end of values.py
################################################################################
# Helper functions dealing with memory blocks.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, dim):
  return np.ctypeslib.as_array(x, shape=dim)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset, bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)
def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.exit(exitcode)
### start of tuning.py
###
### Reading the .tuning file.

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

### end of tuning.py
# Scalar functions.

import numpy as np
import struct

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_round64(x):
  return np.round(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_round32(x):
  return np.round(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])
class image_conv:
  entry_points = {"main": (["i32", "[][][]u8", "[][]f32"], ["[][][]u8"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width", 32),
     ("AMD Accelerated Parallel Processing", cl.device_type.GPU, "lockstep_width",
      64), ("", cl.device_type.GPU, "lockstep_width", 1), ("", cl.device_type.GPU,
                                                           "num_groups", 256), ("",
                                                                                cl.device_type.GPU,
                                                                                "group_size",
                                                                                256),
     ("", cl.device_type.GPU, "tile_size", 32), ("", cl.device_type.GPU,
                                                 "threshold", 32768), ("",
                                                                       cl.device_type.CPU,
                                                                       "lockstep_width",
                                                                       1), ("",
                                                                            cl.device_type.CPU,
                                                                            "num_groups",
                                                                            "MAX_COMPUTE_UNITS"),
     ("", cl.device_type.CPU, "group_size", 32), ("", cl.device_type.CPU,
                                                  "tile_size", 4), ("",
                                                                    cl.device_type.CPU,
                                                                    "threshold",
                                                                    "MAX_COMPUTE_UNITS")]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i8", "i32", "f32", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"main.group_size_13457": {"class": "group_size", "value": None},
                                        "main.group_size_13493": {"class": "group_size", "value": None},
                                        "main.group_size_13529": {"class": "group_size", "value": None},
                                        "main.group_size_13545": {"class": "group_size", "value": None},
                                        "main.group_size_13571": {"class": "group_size", "value": None},
                                        "main.group_size_13618": {"class": "group_size", "value": None},
                                        "main.group_size_13654": {"class": "group_size", "value": None},
                                        "main.group_size_13670": {"class": "group_size", "value": None},
                                        "main.group_size_13696": {"class": "group_size", "value": None},
                                        "main.group_size_13743": {"class": "group_size", "value": None},
                                        "main.group_size_13779": {"class": "group_size", "value": None},
                                        "main.group_size_13795": {"class": "group_size", "value": None},
                                        "main.group_size_13821": {"class": "group_size", "value": None},
                                        "main.group_size_13854": {"class": "group_size", "value": None}})
    self.map_13463_var = program.map_13463
    self.map_13499_var = program.map_13499
    self.map_13535_var = program.map_13535
    self.map_13551_var = program.map_13551
    self.map_13577_var = program.map_13577
    self.map_13624_var = program.map_13624
    self.map_13660_var = program.map_13660
    self.map_13676_var = program.map_13676
    self.map_13702_var = program.map_13702
    self.map_13749_var = program.map_13749
    self.map_13785_var = program.map_13785
    self.map_13801_var = program.map_13801
    self.map_13827_var = program.map_13827
    self.map_13860_var = program.map_13860
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
  def futhark_main(self, image_mem_sizze_13963, image_mem_13964,
                   kernel_mem_sizze_13965, kernel_mem_13966, sizze_13091,
                   sizze_13092, sizze_13093, sizze_13094, sizze_13095,
                   iterations_13096):
    dim_zzero_13099 = (np.int32(0) == sizze_13091)
    dim_zzero_13100 = (np.int32(0) == sizze_13092)
    dim_zzero_13101 = (np.int32(0) == sizze_13093)
    y_13102 = (dim_zzero_13100 or dim_zzero_13101)
    old_empty_13103 = (dim_zzero_13099 or y_13102)
    new_empty_13104 = (dim_zzero_13099 or dim_zzero_13100)
    both_empty_13105 = (old_empty_13103 and new_empty_13104)
    dim_match_13106 = (np.int32(3) == sizze_13093)
    empty_or_match_13107 = (both_empty_13105 or dim_match_13106)
    empty_or_match_cert_13108 = True
    assert empty_or_match_13107, ("Error at image_conv.fut:35:1-48:29: %s" % ("function arguments of wrong shape",))
    dim_match_13110 = (np.int32(3) == sizze_13094)
    dim_match_13111 = (np.int32(3) == sizze_13095)
    match_13112 = (dim_match_13110 and dim_match_13111)
    empty_or_match_cert_13113 = True
    assert match_13112, ("Error at image_conv.fut:35:1-48:29: %s" % ("function arguments of wrong shape",))
    nesting_sizze_13456 = (sizze_13091 * sizze_13092)
    group_sizze_13458 = self.sizes["main.group_size_13457"]
    y_13459 = (group_sizze_13458 - np.int32(1))
    x_13460 = (nesting_sizze_13456 + y_13459)
    num_groups_13461 = squot32(x_13460, group_sizze_13458)
    num_threads_13462 = (group_sizze_13458 * num_groups_13461)
    binop_x_13969 = sext_i32_i64(nesting_sizze_13456)
    bytes_13967 = (np.int64(4) * binop_x_13969)
    mem_13970 = opencl_alloc(self, bytes_13967, "mem_13970")
    mem_13974 = opencl_alloc(self, bytes_13967, "mem_13974")
    mem_13978 = opencl_alloc(self, bytes_13967, "mem_13978")
    if ((1 * (np.long(num_groups_13461) * np.long(group_sizze_13458))) != 0):
      self.map_13463_var.set_args(np.int32(sizze_13091), np.int32(sizze_13092),
                                  np.int32(sizze_13093), image_mem_13964,
                                  mem_13970, mem_13974, mem_13978)
      cl.enqueue_nd_range_kernel(self.queue, self.map_13463_var,
                                 ((np.long(num_groups_13461) * np.long(group_sizze_13458)),),
                                 (np.long(group_sizze_13458),))
      if synchronous:
        self.queue.finish()
    group_sizze_13494 = self.sizes["main.group_size_13493"]
    y_13495 = (group_sizze_13494 - np.int32(1))
    group_sizze_13530 = self.sizes["main.group_size_13529"]
    y_13531 = (group_sizze_13530 - np.int32(1))
    group_sizze_13546 = self.sizes["main.group_size_13545"]
    y_13547 = (group_sizze_13546 - np.int32(1))
    group_sizze_13572 = self.sizes["main.group_size_13571"]
    y_13573 = (group_sizze_13572 - np.int32(1))
    group_sizze_13619 = self.sizes["main.group_size_13618"]
    y_13620 = (group_sizze_13619 - np.int32(1))
    group_sizze_13655 = self.sizes["main.group_size_13654"]
    y_13656 = (group_sizze_13655 - np.int32(1))
    group_sizze_13671 = self.sizes["main.group_size_13670"]
    y_13672 = (group_sizze_13671 - np.int32(1))
    group_sizze_13697 = self.sizes["main.group_size_13696"]
    y_13698 = (group_sizze_13697 - np.int32(1))
    group_sizze_13744 = self.sizes["main.group_size_13743"]
    y_13745 = (group_sizze_13744 - np.int32(1))
    group_sizze_13780 = self.sizes["main.group_size_13779"]
    y_13781 = (group_sizze_13780 - np.int32(1))
    group_sizze_13796 = self.sizes["main.group_size_13795"]
    y_13797 = (group_sizze_13796 - np.int32(1))
    group_sizze_13822 = self.sizes["main.group_size_13821"]
    y_13823 = (group_sizze_13822 - np.int32(1))
    sizze_13141 = sizze_13091
    sizze_13142 = sizze_13092
    sizze_13143 = sizze_13091
    sizze_13144 = sizze_13092
    sizze_13145 = sizze_13091
    sizze_13146 = sizze_13092
    rs_mem_sizze_13979 = bytes_13967
    rs_mem_13980 = mem_13970
    gs_mem_sizze_13981 = bytes_13967
    gs_mem_13982 = mem_13974
    bs_mem_sizze_13983 = bytes_13967
    bs_mem_13984 = mem_13978
    _i_13150 = np.int32(0)
    one_14172 = np.int32(1)
    for counter_14171 in range(iterations_13096):
      range_end_13151 = (np.int32(1) + sizze_13141)
      bounds_invalid_upwards_13152 = slt32(range_end_13151, np.int32(0))
      distance_13153 = (np.int32(1) + range_end_13151)
      if bounds_invalid_upwards_13152:
        num_elems_13154 = np.int32(0)
      else:
        num_elems_13154 = distance_13153
      range_end_13156 = (np.int32(1) + sizze_13142)
      bounds_invalid_upwards_13157 = slt32(range_end_13156, np.int32(0))
      distance_13158 = (np.int32(1) + range_end_13156)
      if bounds_invalid_upwards_13157:
        num_elems_13159 = np.int32(0)
      else:
        num_elems_13159 = distance_13158
      nesting_sizze_13492 = (num_elems_13154 * num_elems_13159)
      x_13496 = (nesting_sizze_13492 + y_13495)
      num_groups_13497 = squot32(x_13496, group_sizze_13494)
      num_threads_13498 = (group_sizze_13494 * num_groups_13497)
      binop_x_13987 = sext_i32_i64(nesting_sizze_13492)
      bytes_13985 = (np.int64(4) * binop_x_13987)
      mem_13988 = opencl_alloc(self, bytes_13985, "mem_13988")
      if ((1 * (np.long(num_groups_13497) * np.long(group_sizze_13494))) != 0):
        self.map_13499_var.set_args(np.int32(sizze_13142),
                                    np.int32(range_end_13151),
                                    np.int32(num_elems_13154),
                                    np.int32(range_end_13156),
                                    np.int32(num_elems_13159), rs_mem_13980,
                                    mem_13988)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13499_var,
                                   ((np.long(num_groups_13497) * np.long(group_sizze_13494)),),
                                   (np.long(group_sizze_13494),))
        if synchronous:
          self.queue.finish()
      range_end_13190 = (num_elems_13159 - np.int32(2))
      bounds_invalid_upwards_13191 = slt32(range_end_13190, np.int32(1))
      distance_upwards_exclusive_13192 = (range_end_13190 - np.int32(1))
      distance_13193 = (np.int32(1) + distance_upwards_exclusive_13192)
      if bounds_invalid_upwards_13191:
        num_elems_13194 = np.int32(0)
      else:
        num_elems_13194 = distance_13193
      flat_dim_13196 = (np.int32(3) * num_elems_13154)
      x_13532 = (num_elems_13194 + y_13531)
      num_groups_13533 = squot32(x_13532, group_sizze_13530)
      num_threads_13534 = (group_sizze_13530 * num_groups_13533)
      convop_x_13990 = (num_elems_13194 * flat_dim_13196)
      binop_x_13991 = sext_i32_i64(convop_x_13990)
      bytes_13989 = (np.int64(4) * binop_x_13991)
      mem_13992 = opencl_alloc(self, bytes_13989, "mem_13992")
      if ((1 * (np.long(num_groups_13533) * np.long(group_sizze_13530))) != 0):
        self.map_13535_var.set_args(np.int32(num_elems_13159),
                                    np.int32(num_elems_13194),
                                    np.int32(flat_dim_13196), mem_13988,
                                    mem_13992)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13535_var,
                                   ((np.long(num_groups_13533) * np.long(group_sizze_13530)),),
                                   (np.long(group_sizze_13530),))
        if synchronous:
          self.queue.finish()
      mem_13988 = None
      x_13204 = (flat_dim_13196 - np.int32(9))
      arg_13205 = (np.int32(3) + x_13204)
      res_13206 = sdiv32(arg_13205, np.int32(3))
      x_13548 = (res_13206 + y_13547)
      num_groups_13549 = squot32(x_13548, group_sizze_13546)
      num_threads_13550 = (group_sizze_13546 * num_groups_13549)
      mem_13996 = opencl_alloc(self, bytes_13989, "mem_13996")
      self.futhark__map_transpose_f32(mem_13996, np.int32(0), mem_13992,
                                      np.int32(0), np.int32(1), num_elems_13194,
                                      flat_dim_13196,
                                      (num_elems_13194 * flat_dim_13196),
                                      (num_elems_13194 * flat_dim_13196))
      mem_13992 = None
      binop_x_13998 = (np.int32(9) * num_elems_13194)
      convop_x_13999 = (res_13206 * binop_x_13998)
      binop_x_14000 = sext_i32_i64(convop_x_13999)
      bytes_13997 = (np.int64(4) * binop_x_14000)
      mem_14001 = opencl_alloc(self, bytes_13997, "mem_14001")
      if ((1 * (np.long(num_groups_13549) * np.long(group_sizze_13546))) != 0):
        self.map_13551_var.set_args(np.int32(num_elems_13194),
                                    np.int32(flat_dim_13196),
                                    np.int32(res_13206), mem_13996, mem_14001)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13551_var,
                                   ((np.long(num_groups_13549) * np.long(group_sizze_13546)),),
                                   (np.long(group_sizze_13546),))
        if synchronous:
          self.queue.finish()
      mem_13996 = None
      range_end_13212 = (res_13206 - np.int32(1))
      bounds_invalid_upwards_13213 = slt32(range_end_13212, np.int32(0))
      distance_13214 = (np.int32(1) + range_end_13212)
      if bounds_invalid_upwards_13213:
        num_elems_13215 = np.int32(0)
      else:
        num_elems_13215 = distance_13214
      range_end_13217 = (num_elems_13194 - np.int32(1))
      bounds_invalid_upwards_13218 = slt32(range_end_13217, np.int32(0))
      distance_13219 = (np.int32(1) + range_end_13217)
      if bounds_invalid_upwards_13218:
        num_elems_13220 = np.int32(0)
      else:
        num_elems_13220 = distance_13219
      nesting_sizze_13570 = (num_elems_13215 * num_elems_13220)
      x_13574 = (nesting_sizze_13570 + y_13573)
      num_groups_13575 = squot32(x_13574, group_sizze_13572)
      num_threads_13576 = (group_sizze_13572 * num_groups_13575)
      binop_x_14003 = (num_elems_13194 * res_13206)
      convop_x_14004 = (np.int32(9) * binop_x_14003)
      binop_x_14005 = sext_i32_i64(convop_x_14004)
      bytes_14002 = (np.int64(4) * binop_x_14005)
      mem_14006 = opencl_alloc(self, bytes_14002, "mem_14006")
      self.futhark__map_transpose_f32(mem_14006, np.int32(0), mem_14001,
                                      np.int32(0), np.int32(1), res_13206,
                                      (num_elems_13194 * np.int32(9)),
                                      ((res_13206 * num_elems_13194) * np.int32(9)),
                                      ((res_13206 * num_elems_13194) * np.int32(9)))
      mem_14001 = None
      binop_x_14008 = (np.int32(9) * res_13206)
      convop_x_14009 = (num_elems_13194 * binop_x_14008)
      binop_x_14010 = sext_i32_i64(convop_x_14009)
      bytes_14007 = (np.int64(4) * binop_x_14010)
      mem_14011 = opencl_alloc(self, bytes_14007, "mem_14011")
      self.futhark__map_transpose_f32(mem_14011, np.int32(0), mem_14006,
                                      np.int32(0), np.int32(1), np.int32(9),
                                      (res_13206 * num_elems_13194),
                                      ((res_13206 * num_elems_13194) * np.int32(9)),
                                      ((res_13206 * num_elems_13194) * np.int32(9)))
      mem_14006 = None
      binop_x_14014 = sext_i32_i64(nesting_sizze_13570)
      bytes_14012 = (np.int64(4) * binop_x_14014)
      mem_14015 = opencl_alloc(self, bytes_14012, "mem_14015")
      if ((1 * (np.long(num_groups_13575) * np.long(group_sizze_13572))) != 0):
        self.map_13577_var.set_args(np.int32(sizze_13095),
                                    np.int32(num_elems_13194),
                                    np.int32(res_13206),
                                    np.int32(num_elems_13215),
                                    np.int32(num_elems_13220), kernel_mem_13966,
                                    mem_14011, mem_14015)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13577_var,
                                   ((np.long(num_groups_13575) * np.long(group_sizze_13572)),),
                                   (np.long(group_sizze_13572),))
        if synchronous:
          self.queue.finish()
      mem_14011 = None
      range_end_13234 = (np.int32(1) + sizze_13143)
      bounds_invalid_upwards_13235 = slt32(range_end_13234, np.int32(0))
      distance_13236 = (np.int32(1) + range_end_13234)
      if bounds_invalid_upwards_13235:
        num_elems_13237 = np.int32(0)
      else:
        num_elems_13237 = distance_13236
      range_end_13239 = (np.int32(1) + sizze_13144)
      bounds_invalid_upwards_13240 = slt32(range_end_13239, np.int32(0))
      distance_13241 = (np.int32(1) + range_end_13239)
      if bounds_invalid_upwards_13240:
        num_elems_13242 = np.int32(0)
      else:
        num_elems_13242 = distance_13241
      nesting_sizze_13617 = (num_elems_13237 * num_elems_13242)
      x_13621 = (nesting_sizze_13617 + y_13620)
      num_groups_13622 = squot32(x_13621, group_sizze_13619)
      num_threads_13623 = (group_sizze_13619 * num_groups_13622)
      binop_x_14018 = sext_i32_i64(nesting_sizze_13617)
      bytes_14016 = (np.int64(4) * binop_x_14018)
      mem_14019 = opencl_alloc(self, bytes_14016, "mem_14019")
      if ((1 * (np.long(num_groups_13622) * np.long(group_sizze_13619))) != 0):
        self.map_13624_var.set_args(np.int32(sizze_13144),
                                    np.int32(range_end_13234),
                                    np.int32(num_elems_13237),
                                    np.int32(range_end_13239),
                                    np.int32(num_elems_13242), gs_mem_13982,
                                    mem_14019)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13624_var,
                                   ((np.long(num_groups_13622) * np.long(group_sizze_13619)),),
                                   (np.long(group_sizze_13619),))
        if synchronous:
          self.queue.finish()
      range_end_13273 = (num_elems_13242 - np.int32(2))
      bounds_invalid_upwards_13274 = slt32(range_end_13273, np.int32(1))
      distance_upwards_exclusive_13275 = (range_end_13273 - np.int32(1))
      distance_13276 = (np.int32(1) + distance_upwards_exclusive_13275)
      if bounds_invalid_upwards_13274:
        num_elems_13277 = np.int32(0)
      else:
        num_elems_13277 = distance_13276
      flat_dim_13279 = (np.int32(3) * num_elems_13237)
      x_13657 = (num_elems_13277 + y_13656)
      num_groups_13658 = squot32(x_13657, group_sizze_13655)
      num_threads_13659 = (group_sizze_13655 * num_groups_13658)
      convop_x_14021 = (num_elems_13277 * flat_dim_13279)
      binop_x_14022 = sext_i32_i64(convop_x_14021)
      bytes_14020 = (np.int64(4) * binop_x_14022)
      mem_14023 = opencl_alloc(self, bytes_14020, "mem_14023")
      if ((1 * (np.long(num_groups_13658) * np.long(group_sizze_13655))) != 0):
        self.map_13660_var.set_args(np.int32(num_elems_13242),
                                    np.int32(num_elems_13277),
                                    np.int32(flat_dim_13279), mem_14019,
                                    mem_14023)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13660_var,
                                   ((np.long(num_groups_13658) * np.long(group_sizze_13655)),),
                                   (np.long(group_sizze_13655),))
        if synchronous:
          self.queue.finish()
      mem_14019 = None
      x_13287 = (flat_dim_13279 - np.int32(9))
      arg_13288 = (np.int32(3) + x_13287)
      res_13289 = sdiv32(arg_13288, np.int32(3))
      x_13673 = (res_13289 + y_13672)
      num_groups_13674 = squot32(x_13673, group_sizze_13671)
      num_threads_13675 = (group_sizze_13671 * num_groups_13674)
      mem_14027 = opencl_alloc(self, bytes_14020, "mem_14027")
      self.futhark__map_transpose_f32(mem_14027, np.int32(0), mem_14023,
                                      np.int32(0), np.int32(1), num_elems_13277,
                                      flat_dim_13279,
                                      (num_elems_13277 * flat_dim_13279),
                                      (num_elems_13277 * flat_dim_13279))
      mem_14023 = None
      binop_x_14029 = (np.int32(9) * num_elems_13277)
      convop_x_14030 = (res_13289 * binop_x_14029)
      binop_x_14031 = sext_i32_i64(convop_x_14030)
      bytes_14028 = (np.int64(4) * binop_x_14031)
      mem_14032 = opencl_alloc(self, bytes_14028, "mem_14032")
      if ((1 * (np.long(num_groups_13674) * np.long(group_sizze_13671))) != 0):
        self.map_13676_var.set_args(np.int32(num_elems_13277),
                                    np.int32(flat_dim_13279),
                                    np.int32(res_13289), mem_14027, mem_14032)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13676_var,
                                   ((np.long(num_groups_13674) * np.long(group_sizze_13671)),),
                                   (np.long(group_sizze_13671),))
        if synchronous:
          self.queue.finish()
      mem_14027 = None
      range_end_13295 = (res_13289 - np.int32(1))
      bounds_invalid_upwards_13296 = slt32(range_end_13295, np.int32(0))
      distance_13297 = (np.int32(1) + range_end_13295)
      if bounds_invalid_upwards_13296:
        num_elems_13298 = np.int32(0)
      else:
        num_elems_13298 = distance_13297
      range_end_13300 = (num_elems_13277 - np.int32(1))
      bounds_invalid_upwards_13301 = slt32(range_end_13300, np.int32(0))
      distance_13302 = (np.int32(1) + range_end_13300)
      if bounds_invalid_upwards_13301:
        num_elems_13303 = np.int32(0)
      else:
        num_elems_13303 = distance_13302
      nesting_sizze_13695 = (num_elems_13298 * num_elems_13303)
      x_13699 = (nesting_sizze_13695 + y_13698)
      num_groups_13700 = squot32(x_13699, group_sizze_13697)
      num_threads_13701 = (group_sizze_13697 * num_groups_13700)
      binop_x_14034 = (num_elems_13277 * res_13289)
      convop_x_14035 = (np.int32(9) * binop_x_14034)
      binop_x_14036 = sext_i32_i64(convop_x_14035)
      bytes_14033 = (np.int64(4) * binop_x_14036)
      mem_14037 = opencl_alloc(self, bytes_14033, "mem_14037")
      self.futhark__map_transpose_f32(mem_14037, np.int32(0), mem_14032,
                                      np.int32(0), np.int32(1), res_13289,
                                      (num_elems_13277 * np.int32(9)),
                                      ((res_13289 * num_elems_13277) * np.int32(9)),
                                      ((res_13289 * num_elems_13277) * np.int32(9)))
      mem_14032 = None
      binop_x_14039 = (np.int32(9) * res_13289)
      convop_x_14040 = (num_elems_13277 * binop_x_14039)
      binop_x_14041 = sext_i32_i64(convop_x_14040)
      bytes_14038 = (np.int64(4) * binop_x_14041)
      mem_14042 = opencl_alloc(self, bytes_14038, "mem_14042")
      self.futhark__map_transpose_f32(mem_14042, np.int32(0), mem_14037,
                                      np.int32(0), np.int32(1), np.int32(9),
                                      (res_13289 * num_elems_13277),
                                      ((res_13289 * num_elems_13277) * np.int32(9)),
                                      ((res_13289 * num_elems_13277) * np.int32(9)))
      mem_14037 = None
      binop_x_14045 = sext_i32_i64(nesting_sizze_13695)
      bytes_14043 = (np.int64(4) * binop_x_14045)
      mem_14046 = opencl_alloc(self, bytes_14043, "mem_14046")
      if ((1 * (np.long(num_groups_13700) * np.long(group_sizze_13697))) != 0):
        self.map_13702_var.set_args(np.int32(sizze_13095),
                                    np.int32(num_elems_13277),
                                    np.int32(res_13289),
                                    np.int32(num_elems_13298),
                                    np.int32(num_elems_13303), kernel_mem_13966,
                                    mem_14042, mem_14046)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13702_var,
                                   ((np.long(num_groups_13700) * np.long(group_sizze_13697)),),
                                   (np.long(group_sizze_13697),))
        if synchronous:
          self.queue.finish()
      mem_14042 = None
      range_end_13317 = (np.int32(1) + sizze_13145)
      bounds_invalid_upwards_13318 = slt32(range_end_13317, np.int32(0))
      distance_13319 = (np.int32(1) + range_end_13317)
      if bounds_invalid_upwards_13318:
        num_elems_13320 = np.int32(0)
      else:
        num_elems_13320 = distance_13319
      range_end_13322 = (np.int32(1) + sizze_13146)
      bounds_invalid_upwards_13323 = slt32(range_end_13322, np.int32(0))
      distance_13324 = (np.int32(1) + range_end_13322)
      if bounds_invalid_upwards_13323:
        num_elems_13325 = np.int32(0)
      else:
        num_elems_13325 = distance_13324
      nesting_sizze_13742 = (num_elems_13320 * num_elems_13325)
      x_13746 = (nesting_sizze_13742 + y_13745)
      num_groups_13747 = squot32(x_13746, group_sizze_13744)
      num_threads_13748 = (group_sizze_13744 * num_groups_13747)
      binop_x_14049 = sext_i32_i64(nesting_sizze_13742)
      bytes_14047 = (np.int64(4) * binop_x_14049)
      mem_14050 = opencl_alloc(self, bytes_14047, "mem_14050")
      if ((1 * (np.long(num_groups_13747) * np.long(group_sizze_13744))) != 0):
        self.map_13749_var.set_args(np.int32(sizze_13146),
                                    np.int32(range_end_13317),
                                    np.int32(num_elems_13320),
                                    np.int32(range_end_13322),
                                    np.int32(num_elems_13325), bs_mem_13984,
                                    mem_14050)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13749_var,
                                   ((np.long(num_groups_13747) * np.long(group_sizze_13744)),),
                                   (np.long(group_sizze_13744),))
        if synchronous:
          self.queue.finish()
      range_end_13356 = (num_elems_13325 - np.int32(2))
      bounds_invalid_upwards_13357 = slt32(range_end_13356, np.int32(1))
      distance_upwards_exclusive_13358 = (range_end_13356 - np.int32(1))
      distance_13359 = (np.int32(1) + distance_upwards_exclusive_13358)
      if bounds_invalid_upwards_13357:
        num_elems_13360 = np.int32(0)
      else:
        num_elems_13360 = distance_13359
      flat_dim_13362 = (np.int32(3) * num_elems_13320)
      x_13782 = (num_elems_13360 + y_13781)
      num_groups_13783 = squot32(x_13782, group_sizze_13780)
      num_threads_13784 = (group_sizze_13780 * num_groups_13783)
      convop_x_14052 = (num_elems_13360 * flat_dim_13362)
      binop_x_14053 = sext_i32_i64(convop_x_14052)
      bytes_14051 = (np.int64(4) * binop_x_14053)
      mem_14054 = opencl_alloc(self, bytes_14051, "mem_14054")
      if ((1 * (np.long(num_groups_13783) * np.long(group_sizze_13780))) != 0):
        self.map_13785_var.set_args(np.int32(num_elems_13325),
                                    np.int32(num_elems_13360),
                                    np.int32(flat_dim_13362), mem_14050,
                                    mem_14054)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13785_var,
                                   ((np.long(num_groups_13783) * np.long(group_sizze_13780)),),
                                   (np.long(group_sizze_13780),))
        if synchronous:
          self.queue.finish()
      mem_14050 = None
      x_13370 = (flat_dim_13362 - np.int32(9))
      arg_13371 = (np.int32(3) + x_13370)
      res_13372 = sdiv32(arg_13371, np.int32(3))
      x_13798 = (res_13372 + y_13797)
      num_groups_13799 = squot32(x_13798, group_sizze_13796)
      num_threads_13800 = (group_sizze_13796 * num_groups_13799)
      mem_14058 = opencl_alloc(self, bytes_14051, "mem_14058")
      self.futhark__map_transpose_f32(mem_14058, np.int32(0), mem_14054,
                                      np.int32(0), np.int32(1), num_elems_13360,
                                      flat_dim_13362,
                                      (num_elems_13360 * flat_dim_13362),
                                      (num_elems_13360 * flat_dim_13362))
      mem_14054 = None
      binop_x_14060 = (np.int32(9) * num_elems_13360)
      convop_x_14061 = (res_13372 * binop_x_14060)
      binop_x_14062 = sext_i32_i64(convop_x_14061)
      bytes_14059 = (np.int64(4) * binop_x_14062)
      mem_14063 = opencl_alloc(self, bytes_14059, "mem_14063")
      if ((1 * (np.long(num_groups_13799) * np.long(group_sizze_13796))) != 0):
        self.map_13801_var.set_args(np.int32(num_elems_13360),
                                    np.int32(flat_dim_13362),
                                    np.int32(res_13372), mem_14058, mem_14063)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13801_var,
                                   ((np.long(num_groups_13799) * np.long(group_sizze_13796)),),
                                   (np.long(group_sizze_13796),))
        if synchronous:
          self.queue.finish()
      mem_14058 = None
      range_end_13378 = (res_13372 - np.int32(1))
      bounds_invalid_upwards_13379 = slt32(range_end_13378, np.int32(0))
      distance_13380 = (np.int32(1) + range_end_13378)
      if bounds_invalid_upwards_13379:
        num_elems_13381 = np.int32(0)
      else:
        num_elems_13381 = distance_13380
      range_end_13383 = (num_elems_13360 - np.int32(1))
      bounds_invalid_upwards_13384 = slt32(range_end_13383, np.int32(0))
      distance_13385 = (np.int32(1) + range_end_13383)
      if bounds_invalid_upwards_13384:
        num_elems_13386 = np.int32(0)
      else:
        num_elems_13386 = distance_13385
      nesting_sizze_13820 = (num_elems_13381 * num_elems_13386)
      x_13824 = (nesting_sizze_13820 + y_13823)
      num_groups_13825 = squot32(x_13824, group_sizze_13822)
      num_threads_13826 = (group_sizze_13822 * num_groups_13825)
      binop_x_14065 = (num_elems_13360 * res_13372)
      convop_x_14066 = (np.int32(9) * binop_x_14065)
      binop_x_14067 = sext_i32_i64(convop_x_14066)
      bytes_14064 = (np.int64(4) * binop_x_14067)
      mem_14068 = opencl_alloc(self, bytes_14064, "mem_14068")
      self.futhark__map_transpose_f32(mem_14068, np.int32(0), mem_14063,
                                      np.int32(0), np.int32(1), res_13372,
                                      (num_elems_13360 * np.int32(9)),
                                      ((res_13372 * num_elems_13360) * np.int32(9)),
                                      ((res_13372 * num_elems_13360) * np.int32(9)))
      mem_14063 = None
      binop_x_14070 = (np.int32(9) * res_13372)
      convop_x_14071 = (num_elems_13360 * binop_x_14070)
      binop_x_14072 = sext_i32_i64(convop_x_14071)
      bytes_14069 = (np.int64(4) * binop_x_14072)
      mem_14073 = opencl_alloc(self, bytes_14069, "mem_14073")
      self.futhark__map_transpose_f32(mem_14073, np.int32(0), mem_14068,
                                      np.int32(0), np.int32(1), np.int32(9),
                                      (res_13372 * num_elems_13360),
                                      ((res_13372 * num_elems_13360) * np.int32(9)),
                                      ((res_13372 * num_elems_13360) * np.int32(9)))
      mem_14068 = None
      binop_x_14076 = sext_i32_i64(nesting_sizze_13820)
      bytes_14074 = (np.int64(4) * binop_x_14076)
      mem_14077 = opencl_alloc(self, bytes_14074, "mem_14077")
      if ((1 * (np.long(num_groups_13825) * np.long(group_sizze_13822))) != 0):
        self.map_13827_var.set_args(np.int32(sizze_13095),
                                    np.int32(num_elems_13360),
                                    np.int32(res_13372),
                                    np.int32(num_elems_13381),
                                    np.int32(num_elems_13386), kernel_mem_13966,
                                    mem_14073, mem_14077)
        cl.enqueue_nd_range_kernel(self.queue, self.map_13827_var,
                                   ((np.long(num_groups_13825) * np.long(group_sizze_13822)),),
                                   (np.long(group_sizze_13822),))
        if synchronous:
          self.queue.finish()
      mem_14073 = None
      sizze_tmp_14117 = num_elems_13215
      sizze_tmp_14118 = num_elems_13220
      sizze_tmp_14119 = num_elems_13298
      sizze_tmp_14120 = num_elems_13303
      sizze_tmp_14121 = num_elems_13381
      sizze_tmp_14122 = num_elems_13386
      rs_mem_sizze_tmp_14123 = bytes_14012
      rs_mem_tmp_14124 = mem_14015
      gs_mem_sizze_tmp_14125 = bytes_14043
      gs_mem_tmp_14126 = mem_14046
      bs_mem_sizze_tmp_14127 = bytes_14074
      bs_mem_tmp_14128 = mem_14077
      sizze_13141 = sizze_tmp_14117
      sizze_13142 = sizze_tmp_14118
      sizze_13143 = sizze_tmp_14119
      sizze_13144 = sizze_tmp_14120
      sizze_13145 = sizze_tmp_14121
      sizze_13146 = sizze_tmp_14122
      rs_mem_sizze_13979 = rs_mem_sizze_tmp_14123
      rs_mem_13980 = rs_mem_tmp_14124
      gs_mem_sizze_13981 = gs_mem_sizze_tmp_14125
      gs_mem_13982 = gs_mem_tmp_14126
      bs_mem_sizze_13983 = bs_mem_sizze_tmp_14127
      bs_mem_13984 = bs_mem_tmp_14128
      _i_13150 += one_14172
    sizze_13132 = sizze_13141
    sizze_13133 = sizze_13142
    sizze_13134 = sizze_13143
    sizze_13135 = sizze_13144
    sizze_13136 = sizze_13145
    sizze_13137 = sizze_13146
    res_mem_sizze_14078 = rs_mem_sizze_13979
    res_mem_14079 = rs_mem_13980
    res_mem_sizze_14080 = gs_mem_sizze_13981
    res_mem_14081 = gs_mem_13982
    res_mem_sizze_14082 = bs_mem_sizze_13983
    res_mem_14083 = bs_mem_13984
    mem_13970 = None
    mem_13974 = None
    mem_13978 = None
    y_13400 = smax32(sizze_13133, sizze_13137)
    sizze_13401 = smax32(sizze_13135, y_13400)
    dim_zzero_13402 = (np.int32(0) == sizze_13132)
    dim_zzero_13403 = (np.int32(0) == sizze_13133)
    old_empty_13404 = (dim_zzero_13402 or dim_zzero_13403)
    dim_zzero_13405 = (np.int32(0) == sizze_13401)
    new_empty_13406 = (dim_zzero_13402 or dim_zzero_13405)
    both_empty_13407 = (old_empty_13404 and new_empty_13406)
    dim_match_13408 = (sizze_13401 == sizze_13133)
    empty_or_match_13409 = (both_empty_13407 or dim_match_13408)
    empty_or_match_cert_13410 = True
    assert empty_or_match_13409, ("Error at image_conv.fut:35:1-48:29 -> image_conv.fut:48:6-29: %s" % ("function arguments of wrong shape",))
    dim_zzero_13412 = (np.int32(0) == sizze_13134)
    dim_zzero_13413 = (np.int32(0) == sizze_13135)
    old_empty_13414 = (dim_zzero_13412 or dim_zzero_13413)
    both_empty_13415 = (new_empty_13406 and old_empty_13414)
    dim_match_13416 = (sizze_13132 == sizze_13134)
    dim_match_13417 = (sizze_13401 == sizze_13135)
    match_13418 = (dim_match_13416 and dim_match_13417)
    empty_or_match_13419 = (both_empty_13415 or match_13418)
    empty_or_match_cert_13420 = True
    assert empty_or_match_13419, ("Error at image_conv.fut:35:1-48:29 -> image_conv.fut:48:6-29: %s" % ("function arguments of wrong shape",))
    dim_zzero_13422 = (np.int32(0) == sizze_13136)
    dim_zzero_13423 = (np.int32(0) == sizze_13137)
    old_empty_13424 = (dim_zzero_13422 or dim_zzero_13423)
    both_empty_13425 = (new_empty_13406 and old_empty_13424)
    dim_match_13426 = (sizze_13132 == sizze_13136)
    dim_match_13427 = (sizze_13401 == sizze_13137)
    match_13428 = (dim_match_13426 and dim_match_13427)
    empty_or_match_13429 = (both_empty_13425 or match_13428)
    empty_or_match_cert_13430 = True
    assert empty_or_match_13429, ("Error at image_conv.fut:35:1-48:29 -> image_conv.fut:48:6-29: %s" % ("function arguments of wrong shape",))
    nesting_sizze_13853 = (sizze_13132 * sizze_13401)
    group_sizze_13855 = self.sizes["main.group_size_13854"]
    y_13856 = (group_sizze_13855 - np.int32(1))
    x_13857 = (nesting_sizze_13853 + y_13856)
    num_groups_13858 = squot32(x_13857, group_sizze_13855)
    num_threads_13859 = (group_sizze_13855 * num_groups_13858)
    convop_x_14088 = (np.int32(3) * nesting_sizze_13853)
    bytes_14086 = sext_i32_i64(convop_x_14088)
    mem_14089 = opencl_alloc(self, bytes_14086, "mem_14089")
    num_threads64_14108 = sext_i32_i64(num_threads_13859)
    total_sizze_14109 = (np.int64(3) * num_threads64_14108)
    mem_14085 = opencl_alloc(self, total_sizze_14109, "mem_14085")
    if ((1 * (np.long(num_groups_13858) * np.long(group_sizze_13855))) != 0):
      self.map_13860_var.set_args(np.int32(sizze_13132), np.int32(sizze_13133),
                                  np.int32(sizze_13135), np.int32(sizze_13137),
                                  np.int32(sizze_13401), res_mem_14079,
                                  res_mem_14081, res_mem_14083, mem_14085,
                                  mem_14089)
      cl.enqueue_nd_range_kernel(self.queue, self.map_13860_var,
                                 ((np.long(num_groups_13858) * np.long(group_sizze_13855)),),
                                 (np.long(group_sizze_13855),))
      if synchronous:
        self.queue.finish()
    res_mem_14079 = None
    res_mem_14081 = None
    res_mem_14083 = None
    mem_14085 = None
    both_empty_13447 = (new_empty_13104 and new_empty_13406)
    dim_match_13448 = (sizze_13091 == sizze_13132)
    dim_match_13449 = (sizze_13092 == sizze_13401)
    match_13450 = (dim_match_13448 and dim_match_13449)
    empty_or_match_13451 = (both_empty_13447 or match_13450)
    empty_or_match_cert_13452 = True
    assert empty_or_match_13451, ("Error at image_conv.fut:35:1-48:29 -> image_conv.fut:35:1-48:29: %s%s%d%s%s%d%s%s%s%s%s" % ("Function return value does not match shape of type ",
                                                                                                                               "[",
                                                                                                                               sizze_13091,
                                                                                                                               "]",
                                                                                                                               "[",
                                                                                                                               sizze_13092,
                                                                                                                               "]",
                                                                                                                               "[",
                                                                                                                               "3",
                                                                                                                               "]",
                                                                                                                               "intrinsics.u8"))
    convop_x_14092 = (np.int32(3) * nesting_sizze_13456)
    bytes_14090 = sext_i32_i64(convop_x_14092)
    mem_14093 = opencl_alloc(self, bytes_14090, "mem_14093")
    if (((sizze_13091 * (sizze_13092 * np.int32(3))) * np.int32(1)) != 0):
      cl.enqueue_copy(self.queue, mem_14093, mem_14089,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(np.int32(0)),
                      byte_count=np.long(((sizze_13091 * (sizze_13092 * np.int32(3))) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    mem_14089 = None
    out_arrsizze_14112 = sizze_13091
    out_arrsizze_14113 = sizze_13092
    out_arrsizze_14114 = np.int32(3)
    out_memsizze_14111 = bytes_14090
    out_mem_14110 = mem_14093
    return (out_memsizze_14111, out_mem_14110, out_arrsizze_14112,
            out_arrsizze_14113, out_arrsizze_14114)
  def futhark__map_transpose_f32(self, destmem_0, destoffset_1, srcmem_2,
                                 srcoffset_3, num_arrays_4, x_elems_5,
                                 y_elems_6, in_elems_7, out_elems_8):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_10 = squot32(np.int32(16), x_elems_5)
      mulx_9 = squot32(np.int32(16), y_elems_6)
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(4))))
        if synchronous:
          self.queue.finish()
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                                                            muly_10) + np.int32(16)) - np.int32(1)),
                                                                                                  np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.map_transpose_f32_low_width_var.set_args(np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.map_transpose_f32_low_width_var,
                                       ((np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                   muly_10) + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              self.queue.finish()
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                  mulx_9) + np.int32(16)) - np.int32(1)),
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                                                                    np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.map_transpose_f32_low_height_var.set_args(np.int32(destoffset_1),
                                                             np.int32(srcoffset_3),
                                                             np.int32(num_arrays_4),
                                                             np.int32(x_elems_5),
                                                             np.int32(y_elems_6),
                                                             np.int32(in_elems_7),
                                                             np.int32(out_elems_8),
                                                             np.int32(mulx_9),
                                                             np.int32(muly_10),
                                                             destmem_0,
                                                             srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_transpose_f32_low_height_var,
                                         ((np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                                     mulx_9) + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                self.queue.finish()
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                        np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.map_transpose_f32_small_var.set_args(np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_small_var,
                                           ((np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                                             np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  self.queue.finish()
            else:
              if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                          np.int32(32))) * np.long(np.int32(32)))) * (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                                                                      np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.map_transpose_f32_var.set_args(np.int32(destoffset_1),
                                                    np.int32(srcoffset_3),
                                                    np.int32(num_arrays_4),
                                                    np.int32(x_elems_5),
                                                    np.int32(y_elems_6),
                                                    np.int32(in_elems_7),
                                                    np.int32(out_elems_8),
                                                    np.int32(mulx_9),
                                                    np.int32(muly_10),
                                                    destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_var,
                                           ((np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  self.queue.finish()
    return ()
  def main(self, iterations_13096_ext, image_mem_13964_ext,
           kernel_mem_13966_ext):
    try:
      iterations_13096 = np.int32(ct.c_int32(iterations_13096_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(iterations_13096_ext),
                                                                                                                            iterations_13096_ext))
    try:
      assert ((type(image_mem_13964_ext) in [np.ndarray,
                                             cl.array.Array]) and (image_mem_13964_ext.dtype == np.uint8)), "Parameter has unexpected type"
      sizze_13091 = np.int32(image_mem_13964_ext.shape[0])
      sizze_13092 = np.int32(image_mem_13964_ext.shape[1])
      sizze_13093 = np.int32(image_mem_13964_ext.shape[2])
      image_mem_sizze_13963 = np.int64(image_mem_13964_ext.nbytes)
      if (type(image_mem_13964_ext) == cl.array.Array):
        image_mem_13964 = image_mem_13964_ext.data
      else:
        image_mem_13964 = opencl_alloc(self, image_mem_sizze_13963,
                                       "image_mem_13964")
        if (image_mem_sizze_13963 != 0):
          cl.enqueue_copy(self.queue, image_mem_13964,
                          normaliseArray(image_mem_13964_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]u8",
                                                                                                                            type(image_mem_13964_ext),
                                                                                                                            image_mem_13964_ext))
    try:
      assert ((type(kernel_mem_13966_ext) in [np.ndarray,
                                              cl.array.Array]) and (kernel_mem_13966_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_13094 = np.int32(kernel_mem_13966_ext.shape[0])
      sizze_13095 = np.int32(kernel_mem_13966_ext.shape[1])
      kernel_mem_sizze_13965 = np.int64(kernel_mem_13966_ext.nbytes)
      if (type(kernel_mem_13966_ext) == cl.array.Array):
        kernel_mem_13966 = kernel_mem_13966_ext.data
      else:
        kernel_mem_13966 = opencl_alloc(self, kernel_mem_sizze_13965,
                                        "kernel_mem_13966")
        if (kernel_mem_sizze_13965 != 0):
          cl.enqueue_copy(self.queue, kernel_mem_13966,
                          normaliseArray(kernel_mem_13966_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(kernel_mem_13966_ext),
                                                                                                                            kernel_mem_13966_ext))
    (out_memsizze_14111, out_mem_14110, out_arrsizze_14112, out_arrsizze_14113,
     out_arrsizze_14114) = self.futhark_main(image_mem_sizze_13963,
                                             image_mem_13964,
                                             kernel_mem_sizze_13965,
                                             kernel_mem_13966, sizze_13091,
                                             sizze_13092, sizze_13093,
                                             sizze_13094, sizze_13095,
                                             iterations_13096)
    return cl.array.Array(self.queue, (out_arrsizze_14112, out_arrsizze_14113,
                                       out_arrsizze_14114), ct.c_uint8,
                          data=out_mem_14110)