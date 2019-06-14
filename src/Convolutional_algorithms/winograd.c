/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


/*
 * Initialisation
*/

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path);
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
struct futhark_context
*futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                        cl_command_queue queue);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0,
                                              int dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
cl_mem futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                 struct futhark_f32_2d *arr);
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0, const
                       struct futhark_f32_2d *in0, const
                       struct futhark_f32_2d *in1);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
/* Crash and burn. */

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

/* Some simple utilities for wall-clock timing.

   The function get_wall_time() returns the wall time in microseconds
   (with an unspecified offset).
*/

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims-1; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    shape[i] = 0;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

// Reading little-endian byte sequences.  On big-endian hosts, we flip
// the resulting bytes.

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_2byte(void* dest) {
  uint16_t x;
  int num_elems_read = fread(&x, 2, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  *(uint16_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_4byte(void* dest) {
  uint32_t x;
  int num_elems_read = fread(&x, 4, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  *(uint32_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_8byte(void* dest) {
  uint64_t x;
  int num_elems_read = fread(&x, 8, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  *(uint64_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int write_byte(void* dest) {
  int num_elems_written = fwrite(dest, 1, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_2byte(void* dest) {
  uint16_t x = *(uint16_t*)dest;
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  int num_elems_written = fwrite(&x, 2, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_4byte(void* dest) {
  uint32_t x = *(uint32_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  int num_elems_written = fwrite(&x, 4, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_8byte(void* dest) {
  uint64_t x = *(uint64_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  int num_elems_written = fwrite(&x, 8, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
  const writer write_bin; // Write in binary format.
  const bin_reader read_bin; // Read in binary format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = read_le_8byte(&bin_shape);
    if (ret != 0) { panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i); }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    char* elems = (char*) *data;
    for (uint64_t i=0; i<elem_count; i++) {
      char* elem = elems+(i*elem_size);
      for (unsigned int j=0; j<elem_size/2; j++) {
        char head = elem[j];
        int tail_index = elem_size-1-j;
        elem[j] = elem[tail_index];
        elem[tail_index] = head;
      }
    }
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[]");
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  fwrite(shape, sizeof(int64_t), rank, out);

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    return expected_type->read_bin(dest);
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"platform", required_argument, NULL,
                                            7}, {"device", required_argument,
                                                 NULL, 8},
                                           {"default-group-size",
                                            required_argument, NULL, 9},
                                           {"default-num-groups",
                                            required_argument, NULL, 10},
                                           {"default-tile-size",
                                            required_argument, NULL, 11},
                                           {"default-threshold",
                                            required_argument, NULL, 12},
                                           {"dump-opencl", required_argument,
                                            NULL, 13}, {"load-opencl",
                                                        required_argument, NULL,
                                                        14},
                                           {"dump-opencl-binary",
                                            required_argument, NULL, 15},
                                           {"load-opencl-binary",
                                            required_argument, NULL, 16},
                                           {"print-sizes", no_argument, NULL,
                                            17}, {"size", required_argument,
                                                  NULL, 18}, {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bp:d:", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e')
            entry_point = optarg;
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'p')
            futhark_context_config_set_platform(cfg, optarg);
        if (ch == 8 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 9)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 11)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 12)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 13)
            futhark_context_config_dump_program_to(cfg, optarg);
        if (ch == 14)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 15)
            futhark_context_config_dump_binary_to(cfg, optarg);
        if (ch == 16)
            futhark_context_config_load_binary_from(cfg, optarg);
        if (ch == 17) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 18) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    panic(1, "Unknown size: %s\n", name);
            } else
                panic(1, "Invalid argument for size option: %s\n", optarg);
        }
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?')
            panic(-1, "Unknown option %s\n", argv[optind - 1]);
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_13176;
    int64_t read_shape_13177[2];
    float *read_arr_13178 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_13178, read_shape_13177, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_13179;
    int64_t read_shape_13180[2];
    float *read_arr_13181 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_13181, read_shape_13180, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_13182;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_13176 = futhark_new_f32_2d(ctx, read_arr_13178,
                                                      read_shape_13177[0],
                                                      read_shape_13177[1])) !=
            0);
        assert((read_value_13179 = futhark_new_f32_2d(ctx, read_arr_13181,
                                                      read_shape_13180[0],
                                                      read_shape_13180[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_13182, read_value_13176,
                               read_value_13179);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_13176) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_13179) == 0);
        assert(futhark_free_f32_2d(ctx, result_13182) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_13176 = futhark_new_f32_2d(ctx, read_arr_13178,
                                                      read_shape_13177[0],
                                                      read_shape_13177[1])) !=
            0);
        assert((read_value_13179 = futhark_new_f32_2d(ctx, read_arr_13181,
                                                      read_shape_13180[0],
                                                      read_shape_13180[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_13182, read_value_13176,
                               read_value_13179);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_13176) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_13179) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_13182) == 0);
        }
    }
    free(read_arr_13178);
    free(read_arr_13181);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_13182)[0] *
                            futhark_shape_f32_2d(ctx, result_13182)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_13182, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_13182), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_13182) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    
    int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
    entry_point_fun *entry_point_fun = NULL;
    
    for (int i = 0; i < num_entry_points; i++) {
        if (strcmp(entry_points[i].name, entry_point) == 0) {
            entry_point_fun = entry_points[i].fun;
            break;
        }
    }
    if (entry_point_fun == NULL) {
        fprintf(stderr,
                "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                entry_point);
        for (int i = 0; i < num_entry_points; i++)
            fprintf(stderr, "%s\n", entry_points[i].name);
        return 1;
    }
    entry_point_fun(ctx);
    if (runtime_file != NULL)
        fclose(runtime_file);
    futhark_debugging_report(ctx);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  lock = lock;
}

#endif

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_SILENCE_DEPRECATION // For macOS.
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif
typedef cl_mem fl_mem_t;
/* Free list management */

/* An entry in the free list.  May be invalid, to avoid having to
   deallocate entries as soon as they are removed.  There is also a
   tag, to help with memory reuse. */
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

/* Remove invalid entries from the free list. */
void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }
  // Now p == l->used.
  l->entries = realloc(l->entries, l->used * sizeof(struct free_list_entry));
  l->capacity = l->used;
}

void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

/* Find and remove a memory block of at least the desired size and
   tag.  Returns 0 on success.  */
int free_list_find(struct free_list *l, const char *tag, size_t *size_out, fl_mem_t *mem_out) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid && l->entries[i].tag == tag) {
      l->entries[i].valid = 0;
      *size_out = l->entries[i].size;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

/* Remove the first block in the free list.  Returns 0 if a block was
   removed, and nonzero if the free list was already empty. */
int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}


/* The simple OpenCL runtime framework used by Futhark. */

#define OPENCL_SUCCEED_FATAL(e) opencl_succeed_fatal(e, #e, __FILE__, __LINE__)
#define OPENCL_SUCCEED_NONFATAL(e) opencl_succeed_nonfatal(e, #e, __FILE__, __LINE__)
// Take care not to override an existing error.
#define OPENCL_SUCCEED_OR_RETURN(e) {             \
    char *error = OPENCL_SUCCEED_NONFATAL(e);     \
    if (error) {                                  \
      if (!ctx->error) {                          \
        ctx->error = error;                       \
        return bad;                               \
      } else {                                    \
        free(error);                              \
      }                                           \
    }                                             \
  }

// OPENCL_SUCCEED_OR_RETURN returns the value of the variable 'bad' in
// scope.  By default, it will be this one.  Create a local variable
// of some other type if needed.  This is a bit of a hack, but it
// saves effort in the code generator.
static const int bad = 1;

struct opencl_config {
  int debugging;
  int logging;
  int preferred_device_num;
  const char *preferred_platform;
  const char *preferred_device;

  const char* dump_program_to;
  const char* load_program_from;
  const char* dump_binary_to;
  const char* load_binary_from;

  size_t default_group_size;
  size_t default_num_groups;
  size_t default_tile_size;
  size_t default_threshold;

  int default_group_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  size_t *size_values;
  const char **size_classes;
};

void opencl_config_init(struct opencl_config *cfg,
                        int num_sizes,
                        const char *size_names[],
                        const char *size_vars[],
                        size_t *size_values,
                        const char *size_classes[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_platform = "";
  cfg->preferred_device = "";
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;
  cfg->dump_binary_to = NULL;
  cfg->load_binary_from = NULL;

  // The following are dummy sizes that mean the concrete defaults
  // will be set during initialisation via hardware-inspection-based
  // heuristics.
  cfg->default_group_size = 0;
  cfg->default_num_groups = 0;
  cfg->default_tile_size = 0;

  cfg->default_threshold = 32*1024;

  cfg->default_group_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

struct opencl_context {
  cl_device_id device;
  cl_context ctx;
  cl_command_queue queue;

  struct opencl_config cfg;

  struct free_list free_list;

  size_t max_group_size;
  size_t max_num_groups;
  size_t max_tile_size;
  size_t max_threshold;

  size_t lockstep_width;
};

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_context*, struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

// Read a file into a NUL-terminated string; returns NULL on error.
static char* slurp_file(const char *filename, size_t *size) {
  char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

static const char* opencl_error_string(unsigned int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed_fatal(unsigned int ret,
                                 const char *call,
                                 const char *file,
                                 int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_succeed_nonfatal(unsigned int ret,
                                     const char *call,
                                     const char *file,
                                     int line) {
  if (ret != CL_SUCCESS) {
    return msgprintf("%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
                     file, line, call, ret, opencl_error_string(ret));
  } else {
    return NULL;
  }
}

void set_preferred_platform(struct opencl_config *cfg, const char *s) {
  cfg->preferred_platform = s;
}

void set_preferred_device(struct opencl_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED_FATAL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED_FATAL(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

static int is_blacklisted(const char *platform_name, const char *device_name,
                          const struct opencl_config *cfg) {
  if (strcmp(cfg->preferred_platform, "") != 0 ||
      strcmp(cfg->preferred_device, "") != 0) {
    return 0;
  } else if (strstr(platform_name, "Apple") != NULL &&
             strstr(device_name, "Intel(R) Core(TM)") != NULL) {
    return 1;
  } else {
    return 0;
  }
}

static struct opencl_device_option get_preferred_device(const struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (!is_blacklisted(device.platform_name, device.device_name, cfg) &&
        strstr(device.platform_name, cfg->preferred_platform) != NULL &&
        strstr(device.device_name, cfg->preferred_device) != NULL &&
        num_device_matches++ == cfg->preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.\n");
  exit(1); // Never reached
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int clBuildProgram_error = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (clBuildProgram_error != CL_SUCCESS &&
      clBuildProgram_error != CL_BUILD_PROGRAM_FAILURE) {
    OPENCL_SUCCEED_FATAL(clBuildProgram_error);
  }

  cl_build_status build_status;
  OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program,
                                             device,
                                             CL_PROGRAM_BUILD_STATUS,
                                             sizeof(cl_build_status),
                                             &build_status,
                                             NULL));

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));

    build_log = malloc(ret_val_size+1);
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

/* Fields in a bitmask indicating which types we must be sure are
   available. */
enum opencl_required_type { OPENCL_F64 = 1 };

// We take as input several strings representing the program, because
// C does not guarantee that the compiler supports particularly large
// literals.  Notably, Visual C has a limit of 2048 characters.  The
// array must be NULL-terminated.
static cl_program setup_opencl_with_command_queue(struct opencl_context *ctx,
                                                  cl_command_queue queue,
                                                  const char *srcs[],
                                                  int required_types) {
  int error;

  ctx->queue = queue;

  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx->ctx, NULL));

  // Fill out the device info.  This is redundant work if we are
  // called from setup_opencl() (which is the common case), but I
  // doubt it matters much.
  struct opencl_device_option device_option;
  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_DEVICE,
                                       sizeof(cl_device_id),
                                       &device_option.device,
                                       NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id),
                                 &device_option.platform,
                                 NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type),
                                 &device_option.device_type,
                                 NULL));
  device_option.platform_name = opencl_platform_info(device_option.platform, CL_PLATFORM_NAME);
  device_option.device_name = opencl_device_info(device_option.device, CL_DEVICE_NAME);

  ctx->device = device_option.device;

  if (required_types & OPENCL_F64) {
    cl_uint supported;
    OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                   sizeof(cl_uint), &supported, NULL));
    if (!supported) {
      panic(1, "Program uses double-precision floats, but this is not supported on the chosen device: %s",
            device_option.device_name);
    }
  }

  size_t max_group_size;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  // Make sure this function is defined.
  post_opencl_setup(ctx, &device_option);

  if (max_group_size < ctx->cfg.default_group_size) {
    if (ctx->cfg.default_group_size_changed) {
      fprintf(stderr, "Note: Device limits default group size to %zu (down from %zu).\n",
              max_group_size, ctx->cfg.default_group_size);
    }
    ctx->cfg.default_group_size = max_group_size;
  }

  if (max_tile_size < ctx->cfg.default_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr, "Note: Device limits default tile size to %zu (down from %zu).\n",
              max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = max_tile_size;
  }

  ctx->max_group_size = max_group_size;
  ctx->max_tile_size = max_tile_size; // No limit.
  ctx->max_threshold = ctx->max_num_groups = 0; // No limit.

  // Now we go through all the sizes, clamp them to the valid range,
  // or set them to the default.
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    size_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    size_t max_value, default_value;
    if (strstr(size_class, "group_size") == size_class) {
      max_value = max_group_size;
      default_value = ctx->cfg.default_group_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = max_group_size; // Futhark assumes this constraint.
      default_value = ctx->cfg.default_num_groups;
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = sqrt(max_group_size);
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      max_value = 0; // No limit.
      default_value = ctx->cfg.default_threshold;
    } else {
      panic(1, "Unknown size class for size '%s': %s\n", size_name, size_class);
    }
    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %d (down from %d)\n",
              size_name, (int)max_value, (int)*size_value);
      *size_value = max_value;
    }
  }

  if (ctx->lockstep_width == 0) {
    ctx->lockstep_width = 1;
  }

  if (ctx->cfg.logging) {
    fprintf(stderr, "Lockstep width: %d\n", (int)ctx->lockstep_width);
    fprintf(stderr, "Default group size: %d\n", (int)ctx->cfg.default_group_size);
    fprintf(stderr, "Default number of groups: %d\n", (int)ctx->cfg.default_num_groups);
  }

  char *fut_opencl_src = NULL;
  size_t src_size = 0;

  // Maybe we have to read OpenCL source from somewhere else (used for debugging).
  if (ctx->cfg.load_program_from != NULL) {
    fut_opencl_src = slurp_file(ctx->cfg.load_program_from, NULL);
    assert(fut_opencl_src != NULL);
  } else {
    // Build the OpenCL program.  First we have to concatenate all the fragments.
    for (const char **src = srcs; src && *src; src++) {
      src_size += strlen(*src);
    }

    fut_opencl_src = malloc(src_size + 1);

    size_t n, i;
    for (i = 0, n = 0; srcs && srcs[i]; i++) {
      strncpy(fut_opencl_src+n, srcs[i], src_size-n);
      n += strlen(srcs[i]);
    }
    fut_opencl_src[src_size] = 0;

  }

  cl_program prog;
  error = CL_SUCCESS;
  const char* src_ptr[] = {fut_opencl_src};

  if (ctx->cfg.dump_program_to != NULL) {
    FILE *f = fopen(ctx->cfg.dump_program_to, "w");
    assert(f != NULL);
    fputs(fut_opencl_src, f);
    fclose(f);
  }

  if (ctx->cfg.load_binary_from == NULL) {
    prog = clCreateProgramWithSource(ctx->ctx, 1, src_ptr, &src_size, &error);
    OPENCL_SUCCEED_FATAL(error);

    int compile_opts_size = 1024;
    for (int i = 0; i < ctx->cfg.num_sizes; i++) {
      compile_opts_size += strlen(ctx->cfg.size_names[i]) + 20;
    }
    char *compile_opts = malloc(compile_opts_size);

    int w = snprintf(compile_opts, compile_opts_size,
                     "-DLOCKSTEP_WIDTH=%d ",
                     (int)ctx->lockstep_width);

    for (int i = 0; i < ctx->cfg.num_sizes; i++) {
      w += snprintf(compile_opts+w, compile_opts_size-w,
                    "-D%s=%d ",
                    ctx->cfg.size_vars[i],
                    (int)ctx->cfg.size_values[i]);
    }

    OPENCL_SUCCEED_FATAL(build_opencl_program(prog, device_option.device, compile_opts));

    free(compile_opts);
  } else {
    size_t binary_size;
    unsigned char *fut_opencl_bin =
      (unsigned char*) slurp_file(ctx->cfg.load_binary_from, &binary_size);
    assert(fut_opencl_src != NULL);
    const unsigned char *binaries[1] = { fut_opencl_bin };
    cl_int status = 0;

    prog = clCreateProgramWithBinary(ctx->ctx, 1, &device_option.device,
                                     &binary_size, binaries,
                                     &status, &error);

    OPENCL_SUCCEED_FATAL(status);
    OPENCL_SUCCEED_FATAL(error);
  }

  free(fut_opencl_src);

  if (ctx->cfg.dump_binary_to != NULL) {
    size_t binary_size;
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                                          sizeof(size_t), &binary_size, NULL));
    unsigned char *binary = malloc(binary_size);
    unsigned char *binaries[1] = { binary };
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                                          sizeof(unsigned char*), binaries, NULL));

    FILE *f = fopen(ctx->cfg.dump_binary_to, "w");
    assert(f != NULL);
    fwrite(binary, sizeof(char), binary_size, f);
    fclose(f);
  }

  return prog;
}

static cl_program setup_opencl(struct opencl_context *ctx,
                               const char *srcs[],
                               int required_types) {

  ctx->lockstep_width = 0; // Real value set later.

  free_list_init(&ctx->free_list);

  struct opencl_device_option device_option = get_preferred_device(&ctx->cfg);

  if (ctx->cfg.logging) {
    describe_device_option(device_option);
  }

  // Note that NVIDIA's OpenCL requires the platform property
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)device_option.platform,
    0
  };

  cl_int clCreateContext_error;
  ctx->ctx = clCreateContext(properties, 1, &device_option.device, NULL, NULL, &clCreateContext_error);
  OPENCL_SUCCEED_FATAL(clCreateContext_error);

  cl_int clCreateCommandQueue_error;
  cl_command_queue queue = clCreateCommandQueue(ctx->ctx, device_option.device, 0, &clCreateCommandQueue_error);
  OPENCL_SUCCEED_FATAL(clCreateCommandQueue_error);

  return setup_opencl_with_command_queue(ctx, queue, srcs, required_types);
}

// Allocate memory from driver. The problem is that OpenCL may perform
// lazy allocation, so we cannot know whether an allocation succeeded
// until the first time we try to use it.  Hence we immediately
// perform a write to see if the allocation succeeded.  This is slow,
// but the assumption is that this operation will be rare (most things
// will go through the free list).
int opencl_alloc_actual(struct opencl_context *ctx, size_t size, cl_mem *mem_out) {
  int error;
  *mem_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size, NULL, &error);

  if (error != CL_SUCCESS) {
    return error;
  }

  int x = 2;
  error = clEnqueueWriteBuffer(ctx->queue, *mem_out, 1, 0, sizeof(x), &x, 0, NULL, NULL);

  // No need to wait for completion here. clWaitForEvents() cannot
  // return mem object allocation failures. This implies that the
  // buffer is faulted onto the device on enqueue. (Observation by
  // Andreas Kloeckner.)

  return error;
}

int opencl_alloc(struct opencl_context *ctx, size_t min_size, const char *tag, cl_mem *mem_out) {
  assert(min_size >= 0);
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;

  if (free_list_find(&ctx->free_list, tag, &size, mem_out) == 0) {
    // Successfully found a free block.  Is it big enough?
    //
    // FIXME: we might also want to check whether the block is *too
    // big*, to avoid internal fragmentation.  However, this can
    // sharply impact performance on programs where arrays change size
    // frequently.  Fortunately, such allocations are usually fairly
    // short-lived, as they are necessarily within a loop, so the risk
    // of internal fragmentation resulting in an OOM situation is
    // limited.  However, it would be preferable if we could go back
    // and *shrink* oversize allocations when we encounter an OOM
    // condition.  That is technically feasible, since we do not
    // expose OpenCL pointer values directly to the application, but
    // instead rely on a level of indirection.
    if (size >= min_size) {
      return CL_SUCCESS;
    } else {
      // Not just right - free it.
      int error = clReleaseMemObject(*mem_out);
      if (error != CL_SUCCESS) {
        return error;
      }
    }
  }

  // We have to allocate a new block from the driver.  If the
  // allocation does not succeed, then we might be in an out-of-memory
  // situation.  We now start freeing things from the free list until
  // we think we have freed enough that the allocation will succeed.
  // Since we don't know how far the allocation is from fitting, we
  // have to check after every deallocation.  This might be pretty
  // expensive.  Let's hope that this case is hit rarely.

  int error = opencl_alloc_actual(ctx, min_size, mem_out);

  while (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    cl_mem mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      error = clReleaseMemObject(mem);
      if (error != CL_SUCCESS) {
        return error;
      }
    } else {
      break;
    }
    error = opencl_alloc_actual(ctx, min_size, mem_out);
  }

  return error;
}

int opencl_free(struct opencl_context *ctx, cl_mem mem, const char *tag) {
  size_t size;
  cl_mem existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, tag, &size, &existing_mem) == 0) {
    int error = clReleaseMemObject(existing_mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  int error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

  if (error == CL_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return error;
}

int opencl_free_all(struct opencl_context *ctx) {
  cl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    int error = clReleaseMemObject(mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  return CL_SUCCESS;
}

const char *opencl_program[] =
           {"#ifdef cl_clang_storage_class_specifiers\n#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable\n#endif\n__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}",
            "\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inlin",
            "e int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int6",
            "4_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int6",
            "4_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\nstatic inline int8_t sext_i8_i8(int8_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i8_i16(int8_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i8_i32(int8_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i8_i64(int8_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i16_i8(int16_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i16_i16(int16_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i16_i32(int16_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i16_i64(int16_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i32_i8(int32_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i32_i16(int32_t x)\n{\n    return x;\n}\nstatic in",
            "line int32_t sext_i32_i32(int32_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i32_i64(int32_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i64_i8(int64_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i64_i16(int64_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i64_i32(int64_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i64_i64(int64_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i8_i8(uint8_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i8_i16(uint8_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i8_i32(uint8_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i8_i64(uint8_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i16_i8(uint16_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i16_i16(uint16_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i16_i32(uint16_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i16_i64(uint16_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i32_i8(uint32_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i32_i16(uint32_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i32_i32(uint32_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i32_i64(uint32_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i64_i8(uint64_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i64_i16(uint64_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i64_i32(uint64_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i64_i64(uint64_t x)\n{\n    return x;\n}\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return x < y ? x : y;\n}\nstatic inline float fmax32(float x, float y)\n{\n    return x < y ? y : x;\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(floa",
            "t x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return is",
            "nan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#define tile_sizze_12624 (mainzitile_sizze_12623)\n#define tiled_group_sizze_12625 (mainzitile_sizze_12623 * mainzitile_sizze_12623)\n#define tile_sizze_12649 (mainzitile_sizze_12648)\n#define tiled_group_sizze_12650 (mainzitile_sizze_12648 * mainzitile_sizze_12648)\n#define tile_sizze_12674 (mainzitile_sizze_12673)\n#define tiled_group_sizze_12675 (mainzitile_sizze_12673 * mainzitile_sizze_12673)\n#define tile_sizze_12709 (mainzitile_sizze_12708)\n#define tiled_group_sizze_12710 (mainzitile_sizze_12708 * mainzitile_sizze_12708)\n#define tile_sizze_12744 (mainzitile_sizze_12743)\n#define tiled_group_sizze_12745 (mainzitile_sizze_12743 * mainzitile_sizze_12743)\n#define tile_sizze_12779 (mainzitile_sizze_12778)\n#define tiled_group_sizze_12780 (mainzitile_sizze_12778 * mainzitile_sizze_12778)\n__kernel void copy_13070(int32_t res_12055, int32_t sizze_12064,\n                         int32_t sizze_12065, __global unsigned char *mem_12980,\n                         __global unsigned char *mem_12984)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t copy_gtid_13070;\n    int32_t copy_ltid_13071;\n    int32_t copy_gid_13072;\n    \n    copy_gtid_13070 = get_global_id(0);\n    copy_ltid_13071 = get_local_id(0);\n    copy_gid_13072 = get_group_id(0);\n    if (slt32(copy_gtid_13070, sizze_12064 * sizze_12065)) {\n        *(__global float *) &mem_12984[(squot32(copy_gtid_13070, sizze_12065) *\n                                        sizze_12065 + (copy_gtid_13070 -\n                                                       squot32(copy_gtid_13070,\n                                          ",
            "                     sizze_12065) *\n                                                       sizze_12065)) * 4] =\n            *(__global float *) &mem_12980[(squot32(squot32(copy_gtid_13070,\n                                                            sizze_12065) *\n                                                    sizze_12065 +\n                                                    (copy_gtid_13070 -\n                                                     squot32(copy_gtid_13070,\n                                                             sizze_12065) *\n                                                     sizze_12065), 2 *\n                                                    res_12055 * 2) * (4 *\n                                                                      res_12055) +\n                                            squot32(squot32(copy_gtid_13070,\n                                                            sizze_12065) *\n                                                    sizze_12065 +\n                                                    (copy_gtid_13070 -\n                                                     squot32(copy_gtid_13070,\n                                                             sizze_12065) *\n                                                     sizze_12065) -\n                                                    squot32(squot32(copy_gtid_13070,\n                                                                    sizze_12065) *\n                                                            sizze_12065 +\n                                                            (copy_gtid_13070 -\n                                                             squot32(copy_gtid_13070,\n                                                                     sizze_12065) *\n                                                             sizze_12065), 2 *\n                                                            res_12055 * 2) *\n                                                    (2 * re",
            "s_12055 * 2) -\n                                                    squot32(squot32(copy_gtid_13070,\n                                                                    sizze_12065) *\n                                                            sizze_12065 +\n                                                            (copy_gtid_13070 -\n                                                             squot32(copy_gtid_13070,\n                                                                     sizze_12065) *\n                                                             sizze_12065) -\n                                                            squot32(squot32(copy_gtid_13070,\n                                                                            sizze_12065) *\n                                                                    sizze_12065 +\n                                                                    (copy_gtid_13070 -\n                                                                     squot32(copy_gtid_13070,\n                                                                             sizze_12065) *\n                                                                     sizze_12065),\n                                                                    2 *\n                                                                    res_12055 *\n                                                                    2) * (2 *\n                                                                          res_12055 *\n                                                                          2),\n                                                            res_12055 * 2) *\n                                                    (res_12055 * 2), 2) * 4 +\n                                            squot32(squot32(copy_gtid_13070,\n                                                            sizze_12065) *\n                                                    sizze_12065 +\n                             ",
            "                       (copy_gtid_13070 -\n                                                     squot32(copy_gtid_13070,\n                                                             sizze_12065) *\n                                                     sizze_12065) -\n                                                    squot32(squot32(copy_gtid_13070,\n                                                                    sizze_12065) *\n                                                            sizze_12065 +\n                                                            (copy_gtid_13070 -\n                                                             squot32(copy_gtid_13070,\n                                                                     sizze_12065) *\n                                                             sizze_12065), 2 *\n                                                            res_12055 * 2) *\n                                                    (2 * res_12055 * 2),\n                                                    res_12055 * 2) * 2 +\n                                            (squot32(copy_gtid_13070,\n                                                     sizze_12065) *\n                                             sizze_12065 + (copy_gtid_13070 -\n                                                            squot32(copy_gtid_13070,\n                                                                    sizze_12065) *\n                                                            sizze_12065) -\n                                             squot32(squot32(copy_gtid_13070,\n                                                             sizze_12065) *\n                                                     sizze_12065 +\n                                                     (copy_gtid_13070 -\n                                                      squot32(copy_gtid_13070,\n                                                              sizze_12065) *\n                                   ",
            "                   sizze_12065), 2 *\n                                                     res_12055 * 2) * (2 *\n                                                                       res_12055 *\n                                                                       2) -\n                                             squot32(squot32(copy_gtid_13070,\n                                                             sizze_12065) *\n                                                     sizze_12065 +\n                                                     (copy_gtid_13070 -\n                                                      squot32(copy_gtid_13070,\n                                                              sizze_12065) *\n                                                      sizze_12065) -\n                                                     squot32(squot32(copy_gtid_13070,\n                                                                     sizze_12065) *\n                                                             sizze_12065 +\n                                                             (copy_gtid_13070 -\n                                                              squot32(copy_gtid_13070,\n                                                                      sizze_12065) *\n                                                              sizze_12065), 2 *\n                                                             res_12055 * 2) *\n                                                     (2 * res_12055 * 2),\n                                                     res_12055 * 2) *\n                                             (res_12055 * 2) -\n                                             squot32(squot32(copy_gtid_13070,\n                                                             sizze_12065) *\n                                                     sizze_12065 +\n                                                     (copy_gtid_13070 -\n                                                      sq",
            "uot32(copy_gtid_13070,\n                                                              sizze_12065) *\n                                                      sizze_12065) -\n                                                     squot32(squot32(copy_gtid_13070,\n                                                                     sizze_12065) *\n                                                             sizze_12065 +\n                                                             (copy_gtid_13070 -\n                                                              squot32(copy_gtid_13070,\n                                                                      sizze_12065) *\n                                                              sizze_12065), 2 *\n                                                             res_12055 * 2) *\n                                                     (2 * res_12055 * 2) -\n                                                     squot32(squot32(copy_gtid_13070,\n                                                                     sizze_12065) *\n                                                             sizze_12065 +\n                                                             (copy_gtid_13070 -\n                                                              squot32(copy_gtid_13070,\n                                                                      sizze_12065) *\n                                                              sizze_12065) -\n                                                             squot32(squot32(copy_gtid_13070,\n                                                                             sizze_12065) *\n                                                                     sizze_12065 +\n                                                                     (copy_gtid_13070 -\n                                                                      squot32(copy_gtid_13070,\n                                                                           ",
            "   sizze_12065) *\n                                                                      sizze_12065),\n                                                                     2 *\n                                                                     res_12055 *\n                                                                     2) * (2 *\n                                                                           res_12055 *\n                                                                           2),\n                                                             res_12055 * 2) *\n                                                     (res_12055 * 2), 2) * 2)) *\n                                           4];\n    }\n}\n__kernel void map_12180(int32_t sizze_11974, int32_t range_end_11979,\n                        int32_t num_elems_11982, int32_t range_end_11984,\n                        int32_t num_elems_11987, __global\n                        unsigned char *image_mem_12814, __global\n                        unsigned char *mem_12820)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t wave_sizze_13006;\n    int32_t group_sizze_13007;\n    int32_t gtid_12171;\n    int32_t gtid_12172;\n    int32_t global_tid_12180;\n    int32_t local_tid_12181;\n    int32_t group_id_12182;\n    \n    global_tid_12180 = get_global_id(0);\n    local_tid_12181 = get_local_id(0);\n    group_sizze_13007 = get_local_size(0);\n    wave_sizze_13006 = LOCKSTEP_WIDTH;\n    group_id_12182 = get_group_id(0);\n    gtid_12171 = squot32(global_tid_12180, num_elems_11987);\n    gtid_12172 = global_tid_12180 - squot32(global_tid_12180, num_elems_11987) *\n        num_elems_11987;\n    \n    bool binop_x_12592;\n    bool binop_y_12593;\n    bool index_primexp_12594;\n    bool res_12186;\n    bool x_12187;\n    bool res_12188;\n    bool x_12189;\n    float res_12190;\n    \n    if (slt32(gtid_12171, num_elems_11982) && slt32(gtid_12172,\n                                                    num_ele",
            "ms_11987)) {\n        binop_x_12592 = slt32(0, gtid_12171);\n        binop_y_12593 = slt32(gtid_12171, range_end_11979);\n        index_primexp_12594 = binop_x_12592 && binop_y_12593;\n        res_12186 = slt32(0, gtid_12172);\n        x_12187 = res_12186 && index_primexp_12594;\n        res_12188 = slt32(gtid_12172, range_end_11984);\n        x_12189 = x_12187 && res_12188;\n        if (x_12189) {\n            int32_t i_12191;\n            int32_t i_12192;\n            float res_12193;\n            \n            i_12191 = gtid_12171 - 1;\n            i_12192 = gtid_12172 - 1;\n            res_12193 = *(__global float *) &image_mem_12814[(i_12191 *\n                                                              sizze_11974 +\n                                                              i_12192) * 4];\n            res_12190 = res_12193;\n        } else {\n            res_12190 = 0.0F;\n        }\n    }\n    if (slt32(gtid_12171, num_elems_11982) && slt32(gtid_12172,\n                                                    num_elems_11987)) {\n        *(__global float *) &mem_12820[(gtid_12171 * num_elems_11987 +\n                                        gtid_12172) * 4] = res_12190;\n    }\n}\n__kernel void map_12229(__local volatile int64_t *mem_aligned_0,\n                        __local volatile int64_t *mem_aligned_1, __global\n                        unsigned char *mem_12834, __global\n                        unsigned char *mem_12837, __global\n                        unsigned char *mem_12848)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_12841 = mem_aligned_0;\n    __local volatile char *restrict mem_12845 = mem_aligned_1;\n    int32_t wave_sizze_13017;\n    int32_t group_sizze_13018;\n    int32_t gtid_12220;\n    int32_t gtid_12221;\n    int32_t global_tid_12229;\n    int32_t local_tid_12230;\n    int32_t group_id_12231;\n    int32_t ltid_12651;\n    int32_t ltid_12652;\n    \n    global_tid_12229 = get_global_id(0);\n    loc",
            "al_tid_12230 = get_local_id(0);\n    group_sizze_13018 = get_local_size(0);\n    wave_sizze_13017 = LOCKSTEP_WIDTH;\n    group_id_12231 = get_group_id(0);\n    gtid_12220 = squot32(srem32(global_tid_12229, tile_sizze_12649 *\n                                tile_sizze_12649), tile_sizze_12649) +\n        squot32(squot32(global_tid_12229, tile_sizze_12649 * tile_sizze_12649),\n                squot32(4 + tile_sizze_12649 - 1, tile_sizze_12649)) *\n        tile_sizze_12649;\n    gtid_12221 = srem32(global_tid_12229, tile_sizze_12649 * tile_sizze_12649) -\n        squot32(srem32(global_tid_12229, tile_sizze_12649 * tile_sizze_12649),\n                tile_sizze_12649) * tile_sizze_12649 +\n        (squot32(global_tid_12229, tile_sizze_12649 * tile_sizze_12649) -\n         squot32(squot32(global_tid_12229, tile_sizze_12649 * tile_sizze_12649),\n                 squot32(4 + tile_sizze_12649 - 1, tile_sizze_12649)) *\n         squot32(4 + tile_sizze_12649 - 1, tile_sizze_12649)) *\n        tile_sizze_12649;\n    ltid_12651 = squot32(srem32(global_tid_12229, tile_sizze_12649 *\n                                tile_sizze_12649), tile_sizze_12649);\n    ltid_12652 = srem32(global_tid_12229, tile_sizze_12649 * tile_sizze_12649) -\n        squot32(srem32(global_tid_12229, tile_sizze_12649 * tile_sizze_12649),\n                tile_sizze_12649) * tile_sizze_12649;\n    if (slt32(gtid_12220, 4) && slt32(gtid_12221, 4)) { }\n    \n    float res_12234;\n    float x_12237 = 0.0F;\n    int32_t chunk_sizze_12235;\n    int32_t chunk_offset_12236 = 0;\n    \n    while (slt32(chunk_offset_12236, 3)) {\n        if (slt32(3 - chunk_offset_12236, tile_sizze_12649)) {\n            chunk_sizze_12235 = 3 - chunk_offset_12236;\n        } else {\n            chunk_sizze_12235 = tile_sizze_12649;\n        }\n        for (int32_t comb_iter_13019 = 0; comb_iter_13019 <\n             squot32(tile_sizze_12649 * tile_sizze_12649 +\n                     tiled_group_sizze_12650 - 1, tiled_group_sizze_12650);\n             comb_iter_13019++",
            ") {\n            int32_t cid_12664;\n            int32_t cid_12665;\n            int32_t flat_comb_id_13020 = comb_iter_13019 *\n                    tiled_group_sizze_12650 + local_tid_12230;\n            \n            cid_12664 = squot32(flat_comb_id_13020, tile_sizze_12649);\n            cid_12665 = flat_comb_id_13020 - squot32(flat_comb_id_13020,\n                                                     tile_sizze_12649) *\n                tile_sizze_12649;\n            if ((slt32(cid_12664, tile_sizze_12649) && slt32(cid_12665,\n                                                             chunk_sizze_12235)) &&\n                slt32(gtid_12220, 4)) {\n                float x_chunk_outer_elem_12663 = *(__global\n                                                   float *) &mem_12834[(gtid_12220 *\n                                                                        3 +\n                                                                        chunk_offset_12236 +\n                                                                        ltid_12652) *\n                                                                       4];\n                \n                *(__local float *) &mem_12841[(cid_12664 * tile_sizze_12649 +\n                                               cid_12665) * 4] =\n                    x_chunk_outer_elem_12663;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (slt32(gtid_12220, 4) && slt32(gtid_12221, 4)) { }\n        for (int32_t comb_iter_13021 = 0; comb_iter_13021 <\n             squot32(tile_sizze_12649 * tile_sizze_12649 +\n                     tiled_group_sizze_12650 - 1, tiled_group_sizze_12650);\n             comb_iter_13021++) {\n            int32_t cid_12669;\n            int32_t cid_12670;\n            int32_t flat_comb_id_13022 = comb_iter_13021 *\n                    tiled_group_sizze_12650 + local_tid_12230;\n            \n            cid_12669 = squot32(flat_comb_id_13022, tile_sizze_12649);\n            cid_12670 = flat_comb_id_13022 - squot",
            "32(flat_comb_id_13022,\n                                                     tile_sizze_12649) *\n                tile_sizze_12649;\n            if ((slt32(cid_12669, chunk_sizze_12235) && slt32(cid_12670,\n                                                              tile_sizze_12649)) &&\n                slt32(gtid_12221, 4)) {\n                float x_chunk_outer_elem_12668 = *(__global\n                                                   float *) &mem_12837[(gtid_12221 +\n                                                                        4 *\n                                                                        chunk_offset_12236 +\n                                                                        ltid_12651 *\n                                                                        4) * 4];\n                \n                *(__local float *) &mem_12845[(cid_12669 * tile_sizze_12649 +\n                                               cid_12670) * 4] =\n                    x_chunk_outer_elem_12668;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (slt32(gtid_12220, 4) && slt32(gtid_12221, 4)) { }\n        \n        float res_12240;\n        float sync_12672;\n        float acc_12243 = x_12237;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_12241;\n        \n        groupstream_mapaccum_dummy_chunk_sizze_12241 = 1;\n        if (slt32(gtid_12220, 4) && slt32(gtid_12221, 4)) {\n            if (chunk_sizze_12235 == tile_sizze_12649) {\n                for (int32_t i_12242 = 0; i_12242 < tile_sizze_12649;\n                     i_12242++) {\n                    float x_12246;\n                    float x_12247;\n                    float res_12249;\n                    float res_12251;\n                    \n                    x_12246 = *(__local float *) &mem_12841[(ltid_12651 *\n                                                             tile_sizze_12649 +\n                                                             i_12242) * 4];\n                    x_1",
            "2247 = *(__local float *) &mem_12845[(tile_sizze_12649 *\n                                                             0 + ltid_12652 +\n                                                             tile_sizze_12649 *\n                                                             i_12242 + 0 *\n                                                             tile_sizze_12649) *\n                                                            4];\n                    res_12249 = x_12246 * x_12247;\n                    res_12251 = acc_12243 + res_12249;\n                    \n                    float acc_tmp_13023 = res_12251;\n                    \n                    acc_12243 = acc_tmp_13023;\n                }\n            } else {\n                for (int32_t i_12242 = 0; i_12242 < chunk_sizze_12235;\n                     i_12242++) {\n                    float x_12246;\n                    float x_12247;\n                    float res_12249;\n                    float res_12251;\n                    \n                    x_12246 = *(__local float *) &mem_12841[(ltid_12651 *\n                                                             tile_sizze_12649 +\n                                                             i_12242) * 4];\n                    x_12247 = *(__local float *) &mem_12845[(tile_sizze_12649 *\n                                                             0 + ltid_12652 +\n                                                             tile_sizze_12649 *\n                                                             i_12242 + 0 *\n                                                             tile_sizze_12649) *\n                                                            4];\n                    res_12249 = x_12246 * x_12247;\n                    res_12251 = acc_12243 + res_12249;\n                    \n                    float acc_tmp_13024 = res_12251;\n                    \n                    acc_12243 = acc_tmp_13024;\n                }\n            }\n        }\n        res_12240 = acc_12243;",
            "\n        sync_12672 = res_12240;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_12237 = sync_12672;\n        chunk_offset_12236 += tile_sizze_12649;\n    }\n    res_12234 = x_12237;\n    if (slt32(gtid_12220, 4) && slt32(gtid_12221, 4)) {\n        *(__global float *) &mem_12848[(gtid_12220 * 4 + gtid_12221) * 4] =\n            res_12234;\n    }\n}\n__kernel void map_12272(__local volatile int64_t *mem_aligned_0,\n                        __local volatile int64_t *mem_aligned_1,\n                        int32_t sizze_11976, __global\n                        unsigned char *kernel_mem_12816, __global\n                        unsigned char *mem_12823, __global\n                        unsigned char *mem_12834)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_12827 = mem_aligned_0;\n    __local volatile char *restrict mem_12831 = mem_aligned_1;\n    int32_t wave_sizze_13009;\n    int32_t group_sizze_13010;\n    int32_t gtid_12263;\n    int32_t gtid_12264;\n    int32_t global_tid_12272;\n    int32_t local_tid_12273;\n    int32_t group_id_12274;\n    int32_t ltid_12626;\n    int32_t ltid_12627;\n    \n    global_tid_12272 = get_global_id(0);\n    local_tid_12273 = get_local_id(0);\n    group_sizze_13010 = get_local_size(0);\n    wave_sizze_13009 = LOCKSTEP_WIDTH;\n    group_id_12274 = get_group_id(0);\n    gtid_12263 = squot32(srem32(global_tid_12272, tile_sizze_12624 *\n                                tile_sizze_12624), tile_sizze_12624) +\n        squot32(squot32(global_tid_12272, tile_sizze_12624 * tile_sizze_12624),\n                squot32(3 + tile_sizze_12624 - 1, tile_sizze_12624)) *\n        tile_sizze_12624;\n    gtid_12264 = srem32(global_tid_12272, tile_sizze_12624 * tile_sizze_12624) -\n        squot32(srem32(global_tid_12272, tile_sizze_12624 * tile_sizze_12624),\n                tile_sizze_12624) * tile_sizze_12624 +\n        (squot32(global_tid_12272, tile_sizze_12624 * tile_sizze_12624) -\n         squot32(squot32(",
            "global_tid_12272, tile_sizze_12624 * tile_sizze_12624),\n                 squot32(3 + tile_sizze_12624 - 1, tile_sizze_12624)) *\n         squot32(3 + tile_sizze_12624 - 1, tile_sizze_12624)) *\n        tile_sizze_12624;\n    ltid_12626 = squot32(srem32(global_tid_12272, tile_sizze_12624 *\n                                tile_sizze_12624), tile_sizze_12624);\n    ltid_12627 = srem32(global_tid_12272, tile_sizze_12624 * tile_sizze_12624) -\n        squot32(srem32(global_tid_12272, tile_sizze_12624 * tile_sizze_12624),\n                tile_sizze_12624) * tile_sizze_12624;\n    if (slt32(gtid_12263, 4) && slt32(gtid_12264, 3)) { }\n    \n    float res_12277;\n    float x_12280 = 0.0F;\n    int32_t chunk_sizze_12278;\n    int32_t chunk_offset_12279 = 0;\n    \n    while (slt32(chunk_offset_12279, 3)) {\n        if (slt32(3 - chunk_offset_12279, tile_sizze_12624)) {\n            chunk_sizze_12278 = 3 - chunk_offset_12279;\n        } else {\n            chunk_sizze_12278 = tile_sizze_12624;\n        }\n        for (int32_t comb_iter_13011 = 0; comb_iter_13011 <\n             squot32(tile_sizze_12624 * tile_sizze_12624 +\n                     tiled_group_sizze_12625 - 1, tiled_group_sizze_12625);\n             comb_iter_13011++) {\n            int32_t cid_12639;\n            int32_t cid_12640;\n            int32_t flat_comb_id_13012 = comb_iter_13011 *\n                    tiled_group_sizze_12625 + local_tid_12273;\n            \n            cid_12639 = squot32(flat_comb_id_13012, tile_sizze_12624);\n            cid_12640 = flat_comb_id_13012 - squot32(flat_comb_id_13012,\n                                                     tile_sizze_12624) *\n                tile_sizze_12624;\n            if ((slt32(cid_12639, tile_sizze_12624) && slt32(cid_12640,\n                                                             chunk_sizze_12278)) &&\n                slt32(gtid_12263, 4)) {\n                float x_chunk_outer_elem_12638 = *(__global\n                                                   float *) &mem_12823[(gti",
            "d_12263 *\n                                                                        3 +\n                                                                        chunk_offset_12279 +\n                                                                        ltid_12627) *\n                                                                       4];\n                \n                *(__local float *) &mem_12827[(cid_12639 * tile_sizze_12624 +\n                                               cid_12640) * 4] =\n                    x_chunk_outer_elem_12638;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (slt32(gtid_12263, 4) && slt32(gtid_12264, 3)) { }\n        for (int32_t comb_iter_13013 = 0; comb_iter_13013 <\n             squot32(tile_sizze_12624 * tile_sizze_12624 +\n                     tiled_group_sizze_12625 - 1, tiled_group_sizze_12625);\n             comb_iter_13013++) {\n            int32_t cid_12644;\n            int32_t cid_12645;\n            int32_t flat_comb_id_13014 = comb_iter_13013 *\n                    tiled_group_sizze_12625 + local_tid_12273;\n            \n            cid_12644 = squot32(flat_comb_id_13014, tile_sizze_12624);\n            cid_12645 = flat_comb_id_13014 - squot32(flat_comb_id_13014,\n                                                     tile_sizze_12624) *\n                tile_sizze_12624;\n            if ((slt32(cid_12644, chunk_sizze_12278) && slt32(cid_12645,\n                                                              tile_sizze_12624)) &&\n                slt32(gtid_12264, 3)) {\n                float x_chunk_outer_elem_12643 = *(__global\n                                                   float *) &kernel_mem_12816[(gtid_12264 +\n                                                                               sizze_11976 *\n                                                                               chunk_offset_12279 +\n                                                                               ltid_12626 *\n                   ",
            "                                                            sizze_11976) *\n                                                                              4];\n                \n                *(__local float *) &mem_12831[(cid_12644 * tile_sizze_12624 +\n                                               cid_12645) * 4] =\n                    x_chunk_outer_elem_12643;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (slt32(gtid_12263, 4) && slt32(gtid_12264, 3)) { }\n        \n        float res_12283;\n        float sync_12647;\n        float acc_12286 = x_12280;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_12284;\n        \n        groupstream_mapaccum_dummy_chunk_sizze_12284 = 1;\n        if (slt32(gtid_12263, 4) && slt32(gtid_12264, 3)) {\n            if (chunk_sizze_12278 == tile_sizze_12624) {\n                for (int32_t i_12285 = 0; i_12285 < tile_sizze_12624;\n                     i_12285++) {\n                    float x_12289;\n                    float x_12290;\n                    float res_12292;\n                    float res_12294;\n                    \n                    x_12289 = *(__local float *) &mem_12827[(ltid_12626 *\n                                                             tile_sizze_12624 +\n                                                             i_12285) * 4];\n                    x_12290 = *(__local float *) &mem_12831[(tile_sizze_12624 *\n                                                             0 + ltid_12627 +\n                                                             tile_sizze_12624 *\n                                                             i_12285 + 0 *\n                                                             tile_sizze_12624) *\n                                                            4];\n                    res_12292 = x_12289 * x_12290;\n                    res_12294 = acc_12286 + res_12292;\n                    \n                    float acc_tmp_13015 = res_12294;\n                    \n              ",
            "      acc_12286 = acc_tmp_13015;\n                }\n            } else {\n                for (int32_t i_12285 = 0; i_12285 < chunk_sizze_12278;\n                     i_12285++) {\n                    float x_12289;\n                    float x_12290;\n                    float res_12292;\n                    float res_12294;\n                    \n                    x_12289 = *(__local float *) &mem_12827[(ltid_12626 *\n                                                             tile_sizze_12624 +\n                                                             i_12285) * 4];\n                    x_12290 = *(__local float *) &mem_12831[(tile_sizze_12624 *\n                                                             0 + ltid_12627 +\n                                                             tile_sizze_12624 *\n                                                             i_12285 + 0 *\n                                                             tile_sizze_12624) *\n                                                            4];\n                    res_12292 = x_12289 * x_12290;\n                    res_12294 = acc_12286 + res_12292;\n                    \n                    float acc_tmp_13016 = res_12294;\n                    \n                    acc_12286 = acc_tmp_13016;\n                }\n            }\n        }\n        res_12283 = acc_12286;\n        sync_12647 = res_12283;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_12280 = sync_12647;\n        chunk_offset_12279 += tile_sizze_12624;\n    }\n    res_12277 = x_12280;\n    if (slt32(gtid_12263, 4) && slt32(gtid_12264, 3)) {\n        *(__global float *) &mem_12834[(gtid_12263 * 3 + gtid_12264) * 4] =\n            res_12277;\n    }\n}\n__kernel void map_12337(__local volatile int64_t *mem_aligned_0,\n                        __local volatile int64_t *mem_aligned_1,\n                        int32_t res_12052, int32_t res_12055, __global\n                        unsigned char *mem_12934, __global\n                        unsigned char *mem_1296",
            "0, __global\n                        unsigned char *mem_12974)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_12964 = mem_aligned_0;\n    __local volatile char *restrict mem_12968 = mem_aligned_1;\n    int32_t wave_sizze_13062;\n    int32_t group_sizze_13063;\n    int32_t gtid_12324;\n    int32_t gtid_12325;\n    int32_t gtid_12326;\n    int32_t gtid_12327;\n    int32_t global_tid_12337;\n    int32_t local_tid_12338;\n    int32_t group_id_12339;\n    int32_t ltid_12781;\n    int32_t ltid_12782;\n    int32_t ltid_12783;\n    int32_t ltid_12784;\n    \n    global_tid_12337 = get_global_id(0);\n    local_tid_12338 = get_local_id(0);\n    group_sizze_13063 = get_local_size(0);\n    wave_sizze_13062 = LOCKSTEP_WIDTH;\n    group_id_12339 = get_group_id(0);\n    gtid_12324 = squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                tile_sizze_12779), tile_sizze_12779 *\n                         tile_sizze_12779) + squot32(squot32(global_tid_12337,\n                                                             tile_sizze_12779 *\n                                                             tile_sizze_12779),\n                                                     (res_12055 + 1 - 1) *\n                                                     squot32(2 +\n                                                             tile_sizze_12779 -\n                                                             1,\n                                                             tile_sizze_12779) *\n                                                     squot32(2 +\n                                                             tile_sizze_12779 -\n                                                             1,\n                                                             tile_sizze_12779));\n    gtid_12325 = squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                tile_sizze_12779) -\n                     ",
            "    squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                        tile_sizze_12779), tile_sizze_12779 *\n                                 tile_sizze_12779) * (tile_sizze_12779 *\n                                                      tile_sizze_12779),\n                         tile_sizze_12779 * tile_sizze_12779) +\n        squot32(squot32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779) -\n                squot32(squot32(global_tid_12337, tile_sizze_12779 *\n                                tile_sizze_12779), (res_12055 + 1 - 1) *\n                        squot32(2 + tile_sizze_12779 - 1, tile_sizze_12779) *\n                        squot32(2 + tile_sizze_12779 - 1, tile_sizze_12779)) *\n                ((res_12055 + 1 - 1) * squot32(2 + tile_sizze_12779 - 1,\n                                               tile_sizze_12779) * squot32(2 +\n                                                                           tile_sizze_12779 -\n                                                                           1,\n                                                                           tile_sizze_12779)),\n                squot32(2 + tile_sizze_12779 - 1, tile_sizze_12779) *\n                squot32(2 + tile_sizze_12779 - 1, tile_sizze_12779));\n    gtid_12326 = squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                tile_sizze_12779) -\n                         squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                        tile_sizze_12779), tile_sizze_12779 *\n                                 tile_sizze_12779) * (tile_sizze_12779 *\n                                                      tile_sizze_12779) -\n                         squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                        tile_sizze_12779) -\n                                 squot32(srem32(global_tid_12337,\n                                                tile_sizze_12779 *\n                        ",
            "                        tile_sizze_12779),\n                                         tile_sizze_12779 * tile_sizze_12779) *\n                                 (tile_sizze_12779 * tile_sizze_12779),\n                                 tile_sizze_12779 * tile_sizze_12779) *\n                         (tile_sizze_12779 * tile_sizze_12779),\n                         tile_sizze_12779) + squot32(squot32(global_tid_12337,\n                                                             tile_sizze_12779 *\n                                                             tile_sizze_12779) -\n                                                     squot32(squot32(global_tid_12337,\n                                                                     tile_sizze_12779 *\n                                                                     tile_sizze_12779),\n                                                             (res_12055 + 1 -\n                                                              1) * squot32(2 +\n                                                                           tile_sizze_12779 -\n                                                                           1,\n                                                                           tile_sizze_12779) *\n                                                             squot32(2 +\n                                                                     tile_sizze_12779 -\n                                                                     1,\n                                                                     tile_sizze_12779)) *\n                                                     ((res_12055 + 1 - 1) *\n                                                      squot32(2 +\n                                                              tile_sizze_12779 -\n                                                              1,\n                                                              tile_sizze_12779) *\n                                                 ",
            "     squot32(2 +\n                                                              tile_sizze_12779 -\n                                                              1,\n                                                              tile_sizze_12779)) -\n                                                     squot32(squot32(global_tid_12337,\n                                                                     tile_sizze_12779 *\n                                                                     tile_sizze_12779) -\n                                                             squot32(squot32(global_tid_12337,\n                                                                             tile_sizze_12779 *\n                                                                             tile_sizze_12779),\n                                                                     (res_12055 +\n                                                                      1 - 1) *\n                                                                     squot32(2 +\n                                                                             tile_sizze_12779 -\n                                                                             1,\n                                                                             tile_sizze_12779) *\n                                                                     squot32(2 +\n                                                                             tile_sizze_12779 -\n                                                                             1,\n                                                                             tile_sizze_12779)) *\n                                                             ((res_12055 + 1 -\n                                                               1) * squot32(2 +\n                                                                            tile_sizze_12779 -\n                                                                            1,\n",
            "                                                                            tile_sizze_12779) *\n                                                              squot32(2 +\n                                                                      tile_sizze_12779 -\n                                                                      1,\n                                                                      tile_sizze_12779)),\n                                                             squot32(2 +\n                                                                     tile_sizze_12779 -\n                                                                     1,\n                                                                     tile_sizze_12779) *\n                                                             squot32(2 +\n                                                                     tile_sizze_12779 -\n                                                                     1,\n                                                                     tile_sizze_12779)) *\n                                                     (squot32(2 +\n                                                              tile_sizze_12779 -\n                                                              1,\n                                                              tile_sizze_12779) *\n                                                      squot32(2 +\n                                                              tile_sizze_12779 -\n                                                              1,\n                                                              tile_sizze_12779)),\n                                                     squot32(2 +\n                                                             tile_sizze_12779 -\n                                                             1,\n                                                             tile_sizze_12779)) *\n        tile_sizze_12779;\n    gtid_12327 = srem32(glob",
            "al_tid_12337, tile_sizze_12779 * tile_sizze_12779) -\n        squot32(srem32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779),\n                tile_sizze_12779 * tile_sizze_12779) * (tile_sizze_12779 *\n                                                        tile_sizze_12779) -\n        squot32(srem32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779) -\n                squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                               tile_sizze_12779), tile_sizze_12779 *\n                        tile_sizze_12779) * (tile_sizze_12779 *\n                                             tile_sizze_12779),\n                tile_sizze_12779 * tile_sizze_12779) * (tile_sizze_12779 *\n                                                        tile_sizze_12779) -\n        squot32(srem32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779) -\n                squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                               tile_sizze_12779), tile_sizze_12779 *\n                        tile_sizze_12779) * (tile_sizze_12779 *\n                                             tile_sizze_12779) -\n                squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                               tile_sizze_12779) -\n                        squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                       tile_sizze_12779), tile_sizze_12779 *\n                                tile_sizze_12779) * (tile_sizze_12779 *\n                                                     tile_sizze_12779),\n                        tile_sizze_12779 * tile_sizze_12779) *\n                (tile_sizze_12779 * tile_sizze_12779), tile_sizze_12779) *\n        tile_sizze_12779 + (squot32(global_tid_12337, tile_sizze_12779 *\n                                    tile_sizze_12779) -\n                            squot32(squot32(global_tid_12337, tile_sizze_12779 *\n                                            tile_sizze_12779), (res_12055 + 1 -\n                               ",
            "                                 1) * squot32(2 +\n                                                                             tile_sizze_12779 -\n                                                                             1,\n                                                                             tile_sizze_12779) *\n                                    squot32(2 + tile_sizze_12779 - 1,\n                                            tile_sizze_12779)) * ((res_12055 +\n                                                                   1 - 1) *\n                                                                  squot32(2 +\n                                                                          tile_sizze_12779 -\n                                                                          1,\n                                                                          tile_sizze_12779) *\n                                                                  squot32(2 +\n                                                                          tile_sizze_12779 -\n                                                                          1,\n                                                                          tile_sizze_12779)) -\n                            squot32(squot32(global_tid_12337, tile_sizze_12779 *\n                                            tile_sizze_12779) -\n                                    squot32(squot32(global_tid_12337,\n                                                    tile_sizze_12779 *\n                                                    tile_sizze_12779),\n                                            (res_12055 + 1 - 1) * squot32(2 +\n                                                                          tile_sizze_12779 -\n                                                                          1,\n                                                                          tile_sizze_12779) *\n                                            squot32(2 + tile_sizze_1",
            "2779 - 1,\n                                                    tile_sizze_12779)) *\n                                    ((res_12055 + 1 - 1) * squot32(2 +\n                                                                   tile_sizze_12779 -\n                                                                   1,\n                                                                   tile_sizze_12779) *\n                                     squot32(2 + tile_sizze_12779 - 1,\n                                             tile_sizze_12779)), squot32(2 +\n                                                                         tile_sizze_12779 -\n                                                                         1,\n                                                                         tile_sizze_12779) *\n                                    squot32(2 + tile_sizze_12779 - 1,\n                                            tile_sizze_12779)) * (squot32(2 +\n                                                                          tile_sizze_12779 -\n                                                                          1,\n                                                                          tile_sizze_12779) *\n                                                                  squot32(2 +\n                                                                          tile_sizze_12779 -\n                                                                          1,\n                                                                          tile_sizze_12779)) -\n                            squot32(squot32(global_tid_12337, tile_sizze_12779 *\n                                            tile_sizze_12779) -\n                                    squot32(squot32(global_tid_12337,\n                                                    tile_sizze_12779 *\n                                                    tile_sizze_12779),\n                                            (res_12055 + 1 - 1) * squot32(2 +\n    ",
            "                                                                      tile_sizze_12779 -\n                                                                          1,\n                                                                          tile_sizze_12779) *\n                                            squot32(2 + tile_sizze_12779 - 1,\n                                                    tile_sizze_12779)) *\n                                    ((res_12055 + 1 - 1) * squot32(2 +\n                                                                   tile_sizze_12779 -\n                                                                   1,\n                                                                   tile_sizze_12779) *\n                                     squot32(2 + tile_sizze_12779 - 1,\n                                             tile_sizze_12779)) -\n                                    squot32(squot32(global_tid_12337,\n                                                    tile_sizze_12779 *\n                                                    tile_sizze_12779) -\n                                            squot32(squot32(global_tid_12337,\n                                                            tile_sizze_12779 *\n                                                            tile_sizze_12779),\n                                                    (res_12055 + 1 - 1) *\n                                                    squot32(2 +\n                                                            tile_sizze_12779 -\n                                                            1,\n                                                            tile_sizze_12779) *\n                                                    squot32(2 +\n                                                            tile_sizze_12779 -\n                                                            1,\n                                                            tile_sizze_12779)) *\n                                          ",
            "  ((res_12055 + 1 - 1) * squot32(2 +\n                                                                           tile_sizze_12779 -\n                                                                           1,\n                                                                           tile_sizze_12779) *\n                                             squot32(2 + tile_sizze_12779 - 1,\n                                                     tile_sizze_12779)),\n                                            squot32(2 + tile_sizze_12779 - 1,\n                                                    tile_sizze_12779) *\n                                            squot32(2 + tile_sizze_12779 - 1,\n                                                    tile_sizze_12779)) *\n                                    (squot32(2 + tile_sizze_12779 - 1,\n                                             tile_sizze_12779) * squot32(2 +\n                                                                         tile_sizze_12779 -\n                                                                         1,\n                                                                         tile_sizze_12779)),\n                                    squot32(2 + tile_sizze_12779 - 1,\n                                            tile_sizze_12779)) * squot32(2 +\n                                                                         tile_sizze_12779 -\n                                                                         1,\n                                                                         tile_sizze_12779)) *\n        tile_sizze_12779;\n    ltid_12781 = squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                tile_sizze_12779), tile_sizze_12779 *\n                         tile_sizze_12779);\n    ltid_12782 = squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                tile_sizze_12779) -\n                         squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                  ",
            "                      tile_sizze_12779), tile_sizze_12779 *\n                                 tile_sizze_12779) * (tile_sizze_12779 *\n                                                      tile_sizze_12779),\n                         tile_sizze_12779 * tile_sizze_12779);\n    ltid_12783 = squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                tile_sizze_12779) -\n                         squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                        tile_sizze_12779), tile_sizze_12779 *\n                                 tile_sizze_12779) * (tile_sizze_12779 *\n                                                      tile_sizze_12779) -\n                         squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                        tile_sizze_12779) -\n                                 squot32(srem32(global_tid_12337,\n                                                tile_sizze_12779 *\n                                                tile_sizze_12779),\n                                         tile_sizze_12779 * tile_sizze_12779) *\n                                 (tile_sizze_12779 * tile_sizze_12779),\n                                 tile_sizze_12779 * tile_sizze_12779) *\n                         (tile_sizze_12779 * tile_sizze_12779),\n                         tile_sizze_12779);\n    ltid_12784 = srem32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779) -\n        squot32(srem32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779),\n                tile_sizze_12779 * tile_sizze_12779) * (tile_sizze_12779 *\n                                                        tile_sizze_12779) -\n        squot32(srem32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779) -\n                squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                               tile_sizze_12779), tile_sizze_12779 *\n                        tile_sizze_12779) * (tile_sizze_12779 *\n                                             ti",
            "le_sizze_12779),\n                tile_sizze_12779 * tile_sizze_12779) * (tile_sizze_12779 *\n                                                        tile_sizze_12779) -\n        squot32(srem32(global_tid_12337, tile_sizze_12779 * tile_sizze_12779) -\n                squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                               tile_sizze_12779), tile_sizze_12779 *\n                        tile_sizze_12779) * (tile_sizze_12779 *\n                                             tile_sizze_12779) -\n                squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                               tile_sizze_12779) -\n                        squot32(srem32(global_tid_12337, tile_sizze_12779 *\n                                       tile_sizze_12779), tile_sizze_12779 *\n                                tile_sizze_12779) * (tile_sizze_12779 *\n                                                     tile_sizze_12779),\n                        tile_sizze_12779 * tile_sizze_12779) *\n                (tile_sizze_12779 * tile_sizze_12779), tile_sizze_12779) *\n        tile_sizze_12779;\n    if (((slt32(gtid_12324, res_12052) && slt32(gtid_12325, res_12055)) &&\n         slt32(gtid_12326, 2)) && slt32(gtid_12327, 2)) { }\n    \n    float res_12342;\n    float x_12345 = 0.0F;\n    int32_t chunk_sizze_12343;\n    int32_t chunk_offset_12344 = 0;\n    \n    while (slt32(chunk_offset_12344, 4)) {\n        if (slt32(4 - chunk_offset_12344, tile_sizze_12779)) {\n            chunk_sizze_12343 = 4 - chunk_offset_12344;\n        } else {\n            chunk_sizze_12343 = tile_sizze_12779;\n        }\n        for (int32_t comb_iter_13064 = 0; comb_iter_13064 <\n             squot32(tile_sizze_12779 * tile_sizze_12779 +\n                     tiled_group_sizze_12780 - 1, tiled_group_sizze_12780);\n             comb_iter_13064++) {\n            int32_t cid_12804;\n            int32_t cid_12805;\n            int32_t flat_comb_id_13065 = comb_iter_13064 *\n                    tiled_group_sizze_12780 + local_tid_12",
            "338;\n            \n            cid_12804 = squot32(flat_comb_id_13065, tile_sizze_12779);\n            cid_12805 = flat_comb_id_13065 - squot32(flat_comb_id_13065,\n                                                     tile_sizze_12779) *\n                tile_sizze_12779;\n            if ((slt32(cid_12804, tile_sizze_12779) && slt32(cid_12805,\n                                                             chunk_sizze_12343)) &&\n                slt32(gtid_12326, 2)) {\n                float x_chunk_outer_elem_12803 = *(__global\n                                                   float *) &mem_12960[(gtid_12324 *\n                                                                        (8 *\n                                                                         res_12055) +\n                                                                        gtid_12325 *\n                                                                        8 +\n                                                                        gtid_12326 *\n                                                                        4 +\n                                                                        chunk_offset_12344 +\n                                                                        ltid_12784) *\n                                                                       4];\n                \n                *(__local float *) &mem_12964[(cid_12804 * tile_sizze_12779 +\n                                               cid_12805) * 4] =\n                    x_chunk_outer_elem_12803;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12324, res_12052) && slt32(gtid_12325, res_12055)) &&\n             slt32(gtid_12326, 2)) && slt32(gtid_12327, 2)) { }\n        for (int32_t comb_iter_13066 = 0; comb_iter_13066 <\n             squot32(tile_sizze_12779 * tile_sizze_12779 +\n                     tiled_group_sizze_12780 - 1, tiled_group_sizze_12780);\n             comb_iter_13066++) {\n           ",
            " int32_t cid_12809;\n            int32_t cid_12810;\n            int32_t flat_comb_id_13067 = comb_iter_13066 *\n                    tiled_group_sizze_12780 + local_tid_12338;\n            \n            cid_12809 = squot32(flat_comb_id_13067, tile_sizze_12779);\n            cid_12810 = flat_comb_id_13067 - squot32(flat_comb_id_13067,\n                                                     tile_sizze_12779) *\n                tile_sizze_12779;\n            if ((slt32(cid_12809, chunk_sizze_12343) && slt32(cid_12810,\n                                                              tile_sizze_12779)) &&\n                slt32(gtid_12327, 2)) {\n                float x_chunk_outer_elem_12808 = *(__global\n                                                   float *) &mem_12934[(gtid_12327 +\n                                                                        2 *\n                                                                        chunk_offset_12344 +\n                                                                        ltid_12783 *\n                                                                        2) * 4];\n                \n                *(__local float *) &mem_12968[(cid_12809 * tile_sizze_12779 +\n                                               cid_12810) * 4] =\n                    x_chunk_outer_elem_12808;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12324, res_12052) && slt32(gtid_12325, res_12055)) &&\n             slt32(gtid_12326, 2)) && slt32(gtid_12327, 2)) { }\n        \n        float res_12348;\n        float sync_12812;\n        float acc_12351 = x_12345;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_12349;\n        \n        groupstream_mapaccum_dummy_chunk_sizze_12349 = 1;\n        if (((slt32(gtid_12324, res_12052) && slt32(gtid_12325, res_12055)) &&\n             slt32(gtid_12326, 2)) && slt32(gtid_12327, 2)) {\n            if (chunk_sizze_12343 == tile_sizze_12779) {\n                for (int32_t i_12350 = 0; i_12350 < t",
            "ile_sizze_12779;\n                     i_12350++) {\n                    float x_12354;\n                    float x_12355;\n                    float res_12357;\n                    float res_12359;\n                    \n                    x_12354 = *(__local float *) &mem_12964[(ltid_12783 *\n                                                             tile_sizze_12779 +\n                                                             i_12350) * 4];\n                    x_12355 = *(__local float *) &mem_12968[(tile_sizze_12779 *\n                                                             0 + ltid_12784 +\n                                                             tile_sizze_12779 *\n                                                             i_12350 + 0 *\n                                                             tile_sizze_12779) *\n                                                            4];\n                    res_12357 = x_12354 * x_12355;\n                    res_12359 = acc_12351 + res_12357;\n                    \n                    float acc_tmp_13068 = res_12359;\n                    \n                    acc_12351 = acc_tmp_13068;\n                }\n            } else {\n                for (int32_t i_12350 = 0; i_12350 < chunk_sizze_12343;\n                     i_12350++) {\n                    float x_12354;\n                    float x_12355;\n                    float res_12357;\n                    float res_12359;\n                    \n                    x_12354 = *(__local float *) &mem_12964[(ltid_12783 *\n                                                             tile_sizze_12779 +\n                                                             i_12350) * 4];\n                    x_12355 = *(__local float *) &mem_12968[(tile_sizze_12779 *\n                                                             0 + ltid_12784 +\n                                                             tile_sizze_12779 *\n                                                             i_12350 + ",
            "0 *\n                                                             tile_sizze_12779) *\n                                                            4];\n                    res_12357 = x_12354 * x_12355;\n                    res_12359 = acc_12351 + res_12357;\n                    \n                    float acc_tmp_13069 = res_12359;\n                    \n                    acc_12351 = acc_tmp_13069;\n                }\n            }\n        }\n        res_12348 = acc_12351;\n        sync_12812 = res_12348;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_12345 = sync_12812;\n        chunk_offset_12344 += tile_sizze_12779;\n    }\n    res_12342 = x_12345;\n    if (((slt32(gtid_12324, res_12052) && slt32(gtid_12325, res_12055)) &&\n         slt32(gtid_12326, 2)) && slt32(gtid_12327, 2)) {\n        *(__global float *) &mem_12974[(gtid_12324 * (4 * res_12055) +\n                                        gtid_12325 * 4 + gtid_12326 * 2 +\n                                        gtid_12327) * 4] = res_12342;\n    }\n}\n__kernel void map_12386(__local volatile int64_t *mem_aligned_0,\n                        __local volatile int64_t *mem_aligned_1,\n                        int32_t res_12052, int32_t res_12055, __global\n                        unsigned char *mem_12934, __global\n                        unsigned char *mem_12946, __global\n                        unsigned char *mem_12960)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_12950 = mem_aligned_0;\n    __local volatile char *restrict mem_12954 = mem_aligned_1;\n    int32_t wave_sizze_13054;\n    int32_t group_sizze_13055;\n    int32_t gtid_12373;\n    int32_t gtid_12374;\n    int32_t gtid_12375;\n    int32_t gtid_12376;\n    int32_t global_tid_12386;\n    int32_t local_tid_12387;\n    int32_t group_id_12388;\n    int32_t ltid_12746;\n    int32_t ltid_12747;\n    int32_t ltid_12748;\n    int32_t ltid_12749;\n    \n    global_tid_12386 = get_global_id(0);\n    local_tid_12387 = ",
            "get_local_id(0);\n    group_sizze_13055 = get_local_size(0);\n    wave_sizze_13054 = LOCKSTEP_WIDTH;\n    group_id_12388 = get_group_id(0);\n    gtid_12373 = squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                tile_sizze_12744), tile_sizze_12744 *\n                         tile_sizze_12744) + squot32(squot32(global_tid_12386,\n                                                             tile_sizze_12744 *\n                                                             tile_sizze_12744),\n                                                     (res_12055 + 1 - 1) *\n                                                     squot32(2 +\n                                                             tile_sizze_12744 -\n                                                             1,\n                                                             tile_sizze_12744) *\n                                                     squot32(4 +\n                                                             tile_sizze_12744 -\n                                                             1,\n                                                             tile_sizze_12744));\n    gtid_12374 = squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                tile_sizze_12744) -\n                         squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                        tile_sizze_12744), tile_sizze_12744 *\n                                 tile_sizze_12744) * (tile_sizze_12744 *\n                                                      tile_sizze_12744),\n                         tile_sizze_12744 * tile_sizze_12744) +\n        squot32(squot32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744) -\n                squot32(squot32(global_tid_12386, tile_sizze_12744 *\n                                tile_sizze_12744), (res_12055 + 1 - 1) *\n                        squot32(2 + tile_sizze_12744 - 1, tile_sizze_12744) *\n                        squot32(4 + tile_si",
            "zze_12744 - 1, tile_sizze_12744)) *\n                ((res_12055 + 1 - 1) * squot32(2 + tile_sizze_12744 - 1,\n                                               tile_sizze_12744) * squot32(4 +\n                                                                           tile_sizze_12744 -\n                                                                           1,\n                                                                           tile_sizze_12744)),\n                squot32(2 + tile_sizze_12744 - 1, tile_sizze_12744) *\n                squot32(4 + tile_sizze_12744 - 1, tile_sizze_12744));\n    gtid_12375 = squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                tile_sizze_12744) -\n                         squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                        tile_sizze_12744), tile_sizze_12744 *\n                                 tile_sizze_12744) * (tile_sizze_12744 *\n                                                      tile_sizze_12744) -\n                         squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                        tile_sizze_12744) -\n                                 squot32(srem32(global_tid_12386,\n                                                tile_sizze_12744 *\n                                                tile_sizze_12744),\n                                         tile_sizze_12744 * tile_sizze_12744) *\n                                 (tile_sizze_12744 * tile_sizze_12744),\n                                 tile_sizze_12744 * tile_sizze_12744) *\n                         (tile_sizze_12744 * tile_sizze_12744),\n                         tile_sizze_12744) + squot32(squot32(global_tid_12386,\n                                                             tile_sizze_12744 *\n                                                             tile_sizze_12744) -\n                                                     squot32(squot32(global_tid_12386,\n                              ",
            "                                       tile_sizze_12744 *\n                                                                     tile_sizze_12744),\n                                                             (res_12055 + 1 -\n                                                              1) * squot32(2 +\n                                                                           tile_sizze_12744 -\n                                                                           1,\n                                                                           tile_sizze_12744) *\n                                                             squot32(4 +\n                                                                     tile_sizze_12744 -\n                                                                     1,\n                                                                     tile_sizze_12744)) *\n                                                     ((res_12055 + 1 - 1) *\n                                                      squot32(2 +\n                                                              tile_sizze_12744 -\n                                                              1,\n                                                              tile_sizze_12744) *\n                                                      squot32(4 +\n                                                              tile_sizze_12744 -\n                                                              1,\n                                                              tile_sizze_12744)) -\n                                                     squot32(squot32(global_tid_12386,\n                                                                     tile_sizze_12744 *\n                                                                     tile_sizze_12744) -\n                                                             squot32(squot32(global_tid_12386,\n                                                                             tile_s",
            "izze_12744 *\n                                                                             tile_sizze_12744),\n                                                                     (res_12055 +\n                                                                      1 - 1) *\n                                                                     squot32(2 +\n                                                                             tile_sizze_12744 -\n                                                                             1,\n                                                                             tile_sizze_12744) *\n                                                                     squot32(4 +\n                                                                             tile_sizze_12744 -\n                                                                             1,\n                                                                             tile_sizze_12744)) *\n                                                             ((res_12055 + 1 -\n                                                               1) * squot32(2 +\n                                                                            tile_sizze_12744 -\n                                                                            1,\n                                                                            tile_sizze_12744) *\n                                                              squot32(4 +\n                                                                      tile_sizze_12744 -\n                                                                      1,\n                                                                      tile_sizze_12744)),\n                                                             squot32(2 +\n                                                                     tile_sizze_12744 -\n                                                                     1,\n                                 ",
            "                                    tile_sizze_12744) *\n                                                             squot32(4 +\n                                                                     tile_sizze_12744 -\n                                                                     1,\n                                                                     tile_sizze_12744)) *\n                                                     (squot32(2 +\n                                                              tile_sizze_12744 -\n                                                              1,\n                                                              tile_sizze_12744) *\n                                                      squot32(4 +\n                                                              tile_sizze_12744 -\n                                                              1,\n                                                              tile_sizze_12744)),\n                                                     squot32(4 +\n                                                             tile_sizze_12744 -\n                                                             1,\n                                                             tile_sizze_12744)) *\n        tile_sizze_12744;\n    gtid_12376 = srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744) -\n        squot32(srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744),\n                tile_sizze_12744 * tile_sizze_12744) * (tile_sizze_12744 *\n                                                        tile_sizze_12744) -\n        squot32(srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744) -\n                squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                               tile_sizze_12744), tile_sizze_12744 *\n                        tile_sizze_12744) * (tile_sizze_12744 *\n                                             tile_sizze_12744),\n                tile_sizze_12744 * tile_sizze_12744) * (tile",
            "_sizze_12744 *\n                                                        tile_sizze_12744) -\n        squot32(srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744) -\n                squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                               tile_sizze_12744), tile_sizze_12744 *\n                        tile_sizze_12744) * (tile_sizze_12744 *\n                                             tile_sizze_12744) -\n                squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                               tile_sizze_12744) -\n                        squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                       tile_sizze_12744), tile_sizze_12744 *\n                                tile_sizze_12744) * (tile_sizze_12744 *\n                                                     tile_sizze_12744),\n                        tile_sizze_12744 * tile_sizze_12744) *\n                (tile_sizze_12744 * tile_sizze_12744), tile_sizze_12744) *\n        tile_sizze_12744 + (squot32(global_tid_12386, tile_sizze_12744 *\n                                    tile_sizze_12744) -\n                            squot32(squot32(global_tid_12386, tile_sizze_12744 *\n                                            tile_sizze_12744), (res_12055 + 1 -\n                                                                1) * squot32(2 +\n                                                                             tile_sizze_12744 -\n                                                                             1,\n                                                                             tile_sizze_12744) *\n                                    squot32(4 + tile_sizze_12744 - 1,\n                                            tile_sizze_12744)) * ((res_12055 +\n                                                                   1 - 1) *\n                                                                  squot32(2 +\n                                                              ",
            "            tile_sizze_12744 -\n                                                                          1,\n                                                                          tile_sizze_12744) *\n                                                                  squot32(4 +\n                                                                          tile_sizze_12744 -\n                                                                          1,\n                                                                          tile_sizze_12744)) -\n                            squot32(squot32(global_tid_12386, tile_sizze_12744 *\n                                            tile_sizze_12744) -\n                                    squot32(squot32(global_tid_12386,\n                                                    tile_sizze_12744 *\n                                                    tile_sizze_12744),\n                                            (res_12055 + 1 - 1) * squot32(2 +\n                                                                          tile_sizze_12744 -\n                                                                          1,\n                                                                          tile_sizze_12744) *\n                                            squot32(4 + tile_sizze_12744 - 1,\n                                                    tile_sizze_12744)) *\n                                    ((res_12055 + 1 - 1) * squot32(2 +\n                                                                   tile_sizze_12744 -\n                                                                   1,\n                                                                   tile_sizze_12744) *\n                                     squot32(4 + tile_sizze_12744 - 1,\n                                             tile_sizze_12744)), squot32(2 +\n                                                                         tile_sizze_12744 -\n                                                   ",
            "                      1,\n                                                                         tile_sizze_12744) *\n                                    squot32(4 + tile_sizze_12744 - 1,\n                                            tile_sizze_12744)) * (squot32(2 +\n                                                                          tile_sizze_12744 -\n                                                                          1,\n                                                                          tile_sizze_12744) *\n                                                                  squot32(4 +\n                                                                          tile_sizze_12744 -\n                                                                          1,\n                                                                          tile_sizze_12744)) -\n                            squot32(squot32(global_tid_12386, tile_sizze_12744 *\n                                            tile_sizze_12744) -\n                                    squot32(squot32(global_tid_12386,\n                                                    tile_sizze_12744 *\n                                                    tile_sizze_12744),\n                                            (res_12055 + 1 - 1) * squot32(2 +\n                                                                          tile_sizze_12744 -\n                                                                          1,\n                                                                          tile_sizze_12744) *\n                                            squot32(4 + tile_sizze_12744 - 1,\n                                                    tile_sizze_12744)) *\n                                    ((res_12055 + 1 - 1) * squot32(2 +\n                                                                   tile_sizze_12744 -\n                                                                   1,\n                                                  ",
            "                 tile_sizze_12744) *\n                                     squot32(4 + tile_sizze_12744 - 1,\n                                             tile_sizze_12744)) -\n                                    squot32(squot32(global_tid_12386,\n                                                    tile_sizze_12744 *\n                                                    tile_sizze_12744) -\n                                            squot32(squot32(global_tid_12386,\n                                                            tile_sizze_12744 *\n                                                            tile_sizze_12744),\n                                                    (res_12055 + 1 - 1) *\n                                                    squot32(2 +\n                                                            tile_sizze_12744 -\n                                                            1,\n                                                            tile_sizze_12744) *\n                                                    squot32(4 +\n                                                            tile_sizze_12744 -\n                                                            1,\n                                                            tile_sizze_12744)) *\n                                            ((res_12055 + 1 - 1) * squot32(2 +\n                                                                           tile_sizze_12744 -\n                                                                           1,\n                                                                           tile_sizze_12744) *\n                                             squot32(4 + tile_sizze_12744 - 1,\n                                                     tile_sizze_12744)),\n                                            squot32(2 + tile_sizze_12744 - 1,\n                                                    tile_sizze_12744) *\n                                            squot32(4 + tile_sizze_12744 - 1,\n    ",
            "                                                tile_sizze_12744)) *\n                                    (squot32(2 + tile_sizze_12744 - 1,\n                                             tile_sizze_12744) * squot32(4 +\n                                                                         tile_sizze_12744 -\n                                                                         1,\n                                                                         tile_sizze_12744)),\n                                    squot32(4 + tile_sizze_12744 - 1,\n                                            tile_sizze_12744)) * squot32(4 +\n                                                                         tile_sizze_12744 -\n                                                                         1,\n                                                                         tile_sizze_12744)) *\n        tile_sizze_12744;\n    ltid_12746 = squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                tile_sizze_12744), tile_sizze_12744 *\n                         tile_sizze_12744);\n    ltid_12747 = squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                tile_sizze_12744) -\n                         squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                        tile_sizze_12744), tile_sizze_12744 *\n                                 tile_sizze_12744) * (tile_sizze_12744 *\n                                                      tile_sizze_12744),\n                         tile_sizze_12744 * tile_sizze_12744);\n    ltid_12748 = squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                tile_sizze_12744) -\n                         squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                        tile_sizze_12744), tile_sizze_12744 *\n                                 tile_sizze_12744) * (tile_sizze_12744 *\n                                                      tile_sizze_12744",
            ") -\n                         squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                                        tile_sizze_12744) -\n                                 squot32(srem32(global_tid_12386,\n                                                tile_sizze_12744 *\n                                                tile_sizze_12744),\n                                         tile_sizze_12744 * tile_sizze_12744) *\n                                 (tile_sizze_12744 * tile_sizze_12744),\n                                 tile_sizze_12744 * tile_sizze_12744) *\n                         (tile_sizze_12744 * tile_sizze_12744),\n                         tile_sizze_12744);\n    ltid_12749 = srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744) -\n        squot32(srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744),\n                tile_sizze_12744 * tile_sizze_12744) * (tile_sizze_12744 *\n                                                        tile_sizze_12744) -\n        squot32(srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744) -\n                squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                               tile_sizze_12744), tile_sizze_12744 *\n                        tile_sizze_12744) * (tile_sizze_12744 *\n                                             tile_sizze_12744),\n                tile_sizze_12744 * tile_sizze_12744) * (tile_sizze_12744 *\n                                                        tile_sizze_12744) -\n        squot32(srem32(global_tid_12386, tile_sizze_12744 * tile_sizze_12744) -\n                squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                               tile_sizze_12744), tile_sizze_12744 *\n                        tile_sizze_12744) * (tile_sizze_12744 *\n                                             tile_sizze_12744) -\n                squot32(srem32(global_tid_12386, tile_sizze_12744 *\n                               tile_sizze_12744) -\n                        squot32(srem32(global_tid_12386",
            ", tile_sizze_12744 *\n                                       tile_sizze_12744), tile_sizze_12744 *\n                                tile_sizze_12744) * (tile_sizze_12744 *\n                                                     tile_sizze_12744),\n                        tile_sizze_12744 * tile_sizze_12744) *\n                (tile_sizze_12744 * tile_sizze_12744), tile_sizze_12744) *\n        tile_sizze_12744;\n    if (((slt32(gtid_12373, res_12052) && slt32(gtid_12374, res_12055)) &&\n         slt32(gtid_12375, 2)) && slt32(gtid_12376, 4)) { }\n    \n    float res_12391;\n    float x_12394 = 0.0F;\n    int32_t chunk_sizze_12392;\n    int32_t chunk_offset_12393 = 0;\n    \n    while (slt32(chunk_offset_12393, 4)) {\n        if (slt32(4 - chunk_offset_12393, tile_sizze_12744)) {\n            chunk_sizze_12392 = 4 - chunk_offset_12393;\n        } else {\n            chunk_sizze_12392 = tile_sizze_12744;\n        }\n        for (int32_t comb_iter_13056 = 0; comb_iter_13056 <\n             squot32(tile_sizze_12744 * tile_sizze_12744 +\n                     tiled_group_sizze_12745 - 1, tiled_group_sizze_12745);\n             comb_iter_13056++) {\n            int32_t cid_12769;\n            int32_t cid_12770;\n            int32_t flat_comb_id_13057 = comb_iter_13056 *\n                    tiled_group_sizze_12745 + local_tid_12387;\n            \n            cid_12769 = squot32(flat_comb_id_13057, tile_sizze_12744);\n            cid_12770 = flat_comb_id_13057 - squot32(flat_comb_id_13057,\n                                                     tile_sizze_12744) *\n                tile_sizze_12744;\n            if ((slt32(cid_12769, tile_sizze_12744) && slt32(cid_12770,\n                                                             chunk_sizze_12392)) &&\n                slt32(gtid_12375, 2)) {\n                float x_chunk_outer_elem_12768 = *(__global\n                                                   float *) &mem_12934[(gtid_12375 +\n                                                                        2 *\n  ",
            "                                                                      chunk_offset_12393 +\n                                                                        ltid_12749 *\n                                                                        2) * 4];\n                \n                *(__local float *) &mem_12950[(cid_12769 * tile_sizze_12744 +\n                                               cid_12770) * 4] =\n                    x_chunk_outer_elem_12768;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12373, res_12052) && slt32(gtid_12374, res_12055)) &&\n             slt32(gtid_12375, 2)) && slt32(gtid_12376, 4)) { }\n        for (int32_t comb_iter_13058 = 0; comb_iter_13058 <\n             squot32(tile_sizze_12744 * tile_sizze_12744 +\n                     tiled_group_sizze_12745 - 1, tiled_group_sizze_12745);\n             comb_iter_13058++) {\n            int32_t cid_12774;\n            int32_t cid_12775;\n            int32_t flat_comb_id_13059 = comb_iter_13058 *\n                    tiled_group_sizze_12745 + local_tid_12387;\n            \n            cid_12774 = squot32(flat_comb_id_13059, tile_sizze_12744);\n            cid_12775 = flat_comb_id_13059 - squot32(flat_comb_id_13059,\n                                                     tile_sizze_12744) *\n                tile_sizze_12744;\n            if ((slt32(cid_12774, chunk_sizze_12392) && slt32(cid_12775,\n                                                              tile_sizze_12744)) &&\n                slt32(gtid_12376, 4)) {\n                float x_chunk_outer_elem_12773 = *(__global\n                                                   float *) &mem_12946[(gtid_12373 *\n                                                                        (4 *\n                                                                         res_12055) +\n                                                                        gtid_12374 *\n                                                                 ",
            "       4 +\n                                                                        gtid_12376 +\n                                                                        4 *\n                                                                        res_12055 *\n                                                                        res_12052 *\n                                                                        chunk_offset_12393 +\n                                                                        ltid_12748 *\n                                                                        (4 *\n                                                                         res_12055 *\n                                                                         res_12052)) *\n                                                                       4];\n                \n                *(__local float *) &mem_12954[(cid_12774 * tile_sizze_12744 +\n                                               cid_12775) * 4] =\n                    x_chunk_outer_elem_12773;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12373, res_12052) && slt32(gtid_12374, res_12055)) &&\n             slt32(gtid_12375, 2)) && slt32(gtid_12376, 4)) { }\n        \n        float res_12397;\n        float sync_12777;\n        float acc_12400 = x_12394;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_12398;\n        \n        groupstream_mapaccum_dummy_chunk_sizze_12398 = 1;\n        if (((slt32(gtid_12373, res_12052) && slt32(gtid_12374, res_12055)) &&\n             slt32(gtid_12375, 2)) && slt32(gtid_12376, 4)) {\n            if (chunk_sizze_12392 == tile_sizze_12744) {\n                for (int32_t i_12399 = 0; i_12399 < tile_sizze_12744;\n                     i_12399++) {\n                    float x_12403;\n                    float x_12404;\n                    float res_12406;\n                    float res_12408;\n                    \n                    x_12403 = *(__local float *) &mem_12",
            "950[(ltid_12748 *\n                                                             tile_sizze_12744 +\n                                                             i_12399) * 4];\n                    x_12404 = *(__local float *) &mem_12954[(tile_sizze_12744 *\n                                                             0 + ltid_12749 +\n                                                             tile_sizze_12744 *\n                                                             i_12399 + 0 *\n                                                             tile_sizze_12744) *\n                                                            4];\n                    res_12406 = x_12403 * x_12404;\n                    res_12408 = acc_12400 + res_12406;\n                    \n                    float acc_tmp_13060 = res_12408;\n                    \n                    acc_12400 = acc_tmp_13060;\n                }\n            } else {\n                for (int32_t i_12399 = 0; i_12399 < chunk_sizze_12392;\n                     i_12399++) {\n                    float x_12403;\n                    float x_12404;\n                    float res_12406;\n                    float res_12408;\n                    \n                    x_12403 = *(__local float *) &mem_12950[(ltid_12748 *\n                                                             tile_sizze_12744 +\n                                                             i_12399) * 4];\n                    x_12404 = *(__local float *) &mem_12954[(tile_sizze_12744 *\n                                                             0 + ltid_12749 +\n                                                             tile_sizze_12744 *\n                                                             i_12399 + 0 *\n                                                             tile_sizze_12744) *\n                                                            4];\n                    res_12406 = x_12403 * x_12404;\n                    res_12408 = acc_12400 + res_12406;\n                 ",
            "   \n                    float acc_tmp_13061 = res_12408;\n                    \n                    acc_12400 = acc_tmp_13061;\n                }\n            }\n        }\n        res_12397 = acc_12400;\n        sync_12777 = res_12397;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_12394 = sync_12777;\n        chunk_offset_12393 += tile_sizze_12744;\n    }\n    res_12391 = x_12394;\n    if (((slt32(gtid_12373, res_12052) && slt32(gtid_12374, res_12055)) &&\n         slt32(gtid_12375, 2)) && slt32(gtid_12376, 4)) {\n        *(__global float *) &mem_12960[(gtid_12373 * (8 * res_12055) +\n                                        gtid_12374 * 8 + gtid_12375 * 4 +\n                                        gtid_12376) * 4] = res_12391;\n    }\n}\n__kernel void map_12422(int32_t res_12052, int32_t res_12055, __global\n                        unsigned char *mem_12926, __global\n                        unsigned char *mem_12931)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t wave_sizze_13050;\n    int32_t group_sizze_13051;\n    int32_t gtid_12413;\n    int32_t gtid_12414;\n    int32_t global_tid_12422;\n    int32_t local_tid_12423;\n    int32_t group_id_12424;\n    \n    global_tid_12422 = get_global_id(0);\n    local_tid_12423 = get_local_id(0);\n    group_sizze_13051 = get_local_size(0);\n    wave_sizze_13050 = LOCKSTEP_WIDTH;\n    group_id_12424 = get_group_id(0);\n    gtid_12413 = squot32(global_tid_12422, res_12055);\n    gtid_12414 = global_tid_12422 - squot32(global_tid_12422, res_12055) *\n        res_12055;\n    if (slt32(gtid_12413, res_12052) && slt32(gtid_12414, res_12055)) { }\n    if (slt32(gtid_12413, res_12052) && slt32(gtid_12414, res_12055)) {\n        for (int32_t i_13052 = 0; i_13052 < 4; i_13052++) {\n            for (int32_t i_13053 = 0; i_13053 < 4; i_13053++) {\n                *(__global float *) &mem_12931[(gtid_12413 * res_12055 +\n                                                gtid_12414 + (i_13052 *\n                              ",
            "                                (res_12055 *\n                                                               res_12052 * 4) +\n                                                              i_13053 *\n                                                              (res_12055 *\n                                                               res_12052))) *\n                                               4] = *(__global\n                                                      float *) &mem_12926[(gtid_12413 *\n                                                                           res_12055 +\n                                                                           gtid_12414 +\n                                                                           (i_13053 *\n                                                                            (res_12055 *\n                                                                             res_12052 *\n                                                                             4) +\n                                                                            i_13052 *\n                                                                            (res_12055 *\n                                                                             res_12052))) *\n                                                                          4];\n            }\n        }\n    }\n}\n__kernel void map_12442(int32_t res_12052, int32_t res_12055, __global\n                        unsigned char *mem_12848, __global\n                        unsigned char *mem_12915, __global\n                        unsigned char *mem_12921)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t wave_sizze_13048;\n    int32_t group_sizze_13049;\n    int32_t gtid_12429;\n    int32_t gtid_12430;\n    int32_t gtid_12431;\n    int32_t gtid_12432;\n    int32_t global_tid_12442;\n    int32_t local_tid_12443;\n    int32_t group_id_12444;\n    \n    global_tid_12442 ",
            "= get_global_id(0);\n    local_tid_12443 = get_local_id(0);\n    group_sizze_13049 = get_local_size(0);\n    wave_sizze_13048 = LOCKSTEP_WIDTH;\n    group_id_12444 = get_group_id(0);\n    gtid_12429 = squot32(global_tid_12442, res_12055 * 4 * 4);\n    gtid_12430 = squot32(global_tid_12442 - squot32(global_tid_12442,\n                                                    res_12055 * 4 * 4) *\n                         (res_12055 * 4 * 4), 16);\n    gtid_12431 = squot32(global_tid_12442 - squot32(global_tid_12442,\n                                                    res_12055 * 4 * 4) *\n                         (res_12055 * 4 * 4) - squot32(global_tid_12442 -\n                                                       squot32(global_tid_12442,\n                                                               res_12055 * 4 *\n                                                               4) * (res_12055 *\n                                                                     4 * 4),\n                                                       16) * 16, 4);\n    gtid_12432 = global_tid_12442 - squot32(global_tid_12442, res_12055 * 4 *\n                                            4) * (res_12055 * 4 * 4) -\n        squot32(global_tid_12442 - squot32(global_tid_12442, res_12055 * 4 *\n                                           4) * (res_12055 * 4 * 4), 16) * 16 -\n        squot32(global_tid_12442 - squot32(global_tid_12442, res_12055 * 4 *\n                                           4) * (res_12055 * 4 * 4) -\n                squot32(global_tid_12442 - squot32(global_tid_12442, res_12055 *\n                                                   4 * 4) * (res_12055 * 4 * 4),\n                        16) * 16, 4) * 4;\n    \n    float x_12448;\n    float y_12449;\n    float res_12450;\n    \n    if (((slt32(gtid_12429, res_12052) && slt32(gtid_12430, res_12055)) &&\n         slt32(gtid_12431, 4)) && slt32(gtid_12432, 4)) {\n        x_12448 = *(__global float *) &mem_12915[(gtid_12429 * (16 *\n                                ",
            "                                res_12055) +\n                                                  gtid_12430 * 16 + gtid_12431 *\n                                                  4 + gtid_12432) * 4];\n        y_12449 = *(__global float *) &mem_12848[(gtid_12431 * 4 + gtid_12432) *\n                                                 4];\n        res_12450 = x_12448 * y_12449;\n    }\n    if (((slt32(gtid_12429, res_12052) && slt32(gtid_12430, res_12055)) &&\n         slt32(gtid_12431, 4)) && slt32(gtid_12432, 4)) {\n        *(__global float *) &mem_12921[(gtid_12429 * (16 * res_12055) +\n                                        gtid_12430 * 16 + gtid_12431 * 4 +\n                                        gtid_12432) * 4] = res_12450;\n    }\n}\n__kernel void map_12478(__local volatile int64_t *mem_aligned_0,\n                        __local volatile int64_t *mem_aligned_1,\n                        int32_t res_12052, int32_t res_12055, __global\n                        unsigned char *mem_12875, __global\n                        unsigned char *mem_12901, __global\n                        unsigned char *mem_12915)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_12905 = mem_aligned_0;\n    __local volatile char *restrict mem_12909 = mem_aligned_1;\n    int32_t wave_sizze_13040;\n    int32_t group_sizze_13041;\n    int32_t gtid_12465;\n    int32_t gtid_12466;\n    int32_t gtid_12467;\n    int32_t gtid_12468;\n    int32_t global_tid_12478;\n    int32_t local_tid_12479;\n    int32_t group_id_12480;\n    int32_t ltid_12711;\n    int32_t ltid_12712;\n    int32_t ltid_12713;\n    int32_t ltid_12714;\n    \n    global_tid_12478 = get_global_id(0);\n    local_tid_12479 = get_local_id(0);\n    group_sizze_13041 = get_local_size(0);\n    wave_sizze_13040 = LOCKSTEP_WIDTH;\n    group_id_12480 = get_group_id(0);\n    gtid_12465 = squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                tile_sizze_12709), tile_sizze_12",
            "709 *\n                         tile_sizze_12709) + squot32(squot32(global_tid_12478,\n                                                             tile_sizze_12709 *\n                                                             tile_sizze_12709),\n                                                     (res_12055 + 1 - 1) *\n                                                     squot32(4 +\n                                                             tile_sizze_12709 -\n                                                             1,\n                                                             tile_sizze_12709) *\n                                                     squot32(4 +\n                                                             tile_sizze_12709 -\n                                                             1,\n                                                             tile_sizze_12709));\n    gtid_12466 = squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                tile_sizze_12709) -\n                         squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                        tile_sizze_12709), tile_sizze_12709 *\n                                 tile_sizze_12709) * (tile_sizze_12709 *\n                                                      tile_sizze_12709),\n                         tile_sizze_12709 * tile_sizze_12709) +\n        squot32(squot32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709) -\n                squot32(squot32(global_tid_12478, tile_sizze_12709 *\n                                tile_sizze_12709), (res_12055 + 1 - 1) *\n                        squot32(4 + tile_sizze_12709 - 1, tile_sizze_12709) *\n                        squot32(4 + tile_sizze_12709 - 1, tile_sizze_12709)) *\n                ((res_12055 + 1 - 1) * squot32(4 + tile_sizze_12709 - 1,\n                                               tile_sizze_12709) * squot32(4 +\n                                                                           tile_si",
            "zze_12709 -\n                                                                           1,\n                                                                           tile_sizze_12709)),\n                squot32(4 + tile_sizze_12709 - 1, tile_sizze_12709) *\n                squot32(4 + tile_sizze_12709 - 1, tile_sizze_12709));\n    gtid_12467 = squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                tile_sizze_12709) -\n                         squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                        tile_sizze_12709), tile_sizze_12709 *\n                                 tile_sizze_12709) * (tile_sizze_12709 *\n                                                      tile_sizze_12709) -\n                         squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                        tile_sizze_12709) -\n                                 squot32(srem32(global_tid_12478,\n                                                tile_sizze_12709 *\n                                                tile_sizze_12709),\n                                         tile_sizze_12709 * tile_sizze_12709) *\n                                 (tile_sizze_12709 * tile_sizze_12709),\n                                 tile_sizze_12709 * tile_sizze_12709) *\n                         (tile_sizze_12709 * tile_sizze_12709),\n                         tile_sizze_12709) + squot32(squot32(global_tid_12478,\n                                                             tile_sizze_12709 *\n                                                             tile_sizze_12709) -\n                                                     squot32(squot32(global_tid_12478,\n                                                                     tile_sizze_12709 *\n                                                                     tile_sizze_12709),\n                                                             (res_12055 + 1 -\n                                              ",
            "                1) * squot32(4 +\n                                                                           tile_sizze_12709 -\n                                                                           1,\n                                                                           tile_sizze_12709) *\n                                                             squot32(4 +\n                                                                     tile_sizze_12709 -\n                                                                     1,\n                                                                     tile_sizze_12709)) *\n                                                     ((res_12055 + 1 - 1) *\n                                                      squot32(4 +\n                                                              tile_sizze_12709 -\n                                                              1,\n                                                              tile_sizze_12709) *\n                                                      squot32(4 +\n                                                              tile_sizze_12709 -\n                                                              1,\n                                                              tile_sizze_12709)) -\n                                                     squot32(squot32(global_tid_12478,\n                                                                     tile_sizze_12709 *\n                                                                     tile_sizze_12709) -\n                                                             squot32(squot32(global_tid_12478,\n                                                                             tile_sizze_12709 *\n                                                                             tile_sizze_12709),\n                                                                     (res_12055 +\n                                                                      1 - 1) *\n",
            "                                                                     squot32(4 +\n                                                                             tile_sizze_12709 -\n                                                                             1,\n                                                                             tile_sizze_12709) *\n                                                                     squot32(4 +\n                                                                             tile_sizze_12709 -\n                                                                             1,\n                                                                             tile_sizze_12709)) *\n                                                             ((res_12055 + 1 -\n                                                               1) * squot32(4 +\n                                                                            tile_sizze_12709 -\n                                                                            1,\n                                                                            tile_sizze_12709) *\n                                                              squot32(4 +\n                                                                      tile_sizze_12709 -\n                                                                      1,\n                                                                      tile_sizze_12709)),\n                                                             squot32(4 +\n                                                                     tile_sizze_12709 -\n                                                                     1,\n                                                                     tile_sizze_12709) *\n                                                             squot32(4 +\n                                                                     tile_sizze_12709 -\n                                                     ",
            "                1,\n                                                                     tile_sizze_12709)) *\n                                                     (squot32(4 +\n                                                              tile_sizze_12709 -\n                                                              1,\n                                                              tile_sizze_12709) *\n                                                      squot32(4 +\n                                                              tile_sizze_12709 -\n                                                              1,\n                                                              tile_sizze_12709)),\n                                                     squot32(4 +\n                                                             tile_sizze_12709 -\n                                                             1,\n                                                             tile_sizze_12709)) *\n        tile_sizze_12709;\n    gtid_12468 = srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709) -\n        squot32(srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709),\n                tile_sizze_12709 * tile_sizze_12709) * (tile_sizze_12709 *\n                                                        tile_sizze_12709) -\n        squot32(srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709) -\n                squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                               tile_sizze_12709), tile_sizze_12709 *\n                        tile_sizze_12709) * (tile_sizze_12709 *\n                                             tile_sizze_12709),\n                tile_sizze_12709 * tile_sizze_12709) * (tile_sizze_12709 *\n                                                        tile_sizze_12709) -\n        squot32(srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709) -\n                squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                               ",
            "tile_sizze_12709), tile_sizze_12709 *\n                        tile_sizze_12709) * (tile_sizze_12709 *\n                                             tile_sizze_12709) -\n                squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                               tile_sizze_12709) -\n                        squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                       tile_sizze_12709), tile_sizze_12709 *\n                                tile_sizze_12709) * (tile_sizze_12709 *\n                                                     tile_sizze_12709),\n                        tile_sizze_12709 * tile_sizze_12709) *\n                (tile_sizze_12709 * tile_sizze_12709), tile_sizze_12709) *\n        tile_sizze_12709 + (squot32(global_tid_12478, tile_sizze_12709 *\n                                    tile_sizze_12709) -\n                            squot32(squot32(global_tid_12478, tile_sizze_12709 *\n                                            tile_sizze_12709), (res_12055 + 1 -\n                                                                1) * squot32(4 +\n                                                                             tile_sizze_12709 -\n                                                                             1,\n                                                                             tile_sizze_12709) *\n                                    squot32(4 + tile_sizze_12709 - 1,\n                                            tile_sizze_12709)) * ((res_12055 +\n                                                                   1 - 1) *\n                                                                  squot32(4 +\n                                                                          tile_sizze_12709 -\n                                                                          1,\n                                                                          tile_sizze_12709) *\n                                                                  sq",
            "uot32(4 +\n                                                                          tile_sizze_12709 -\n                                                                          1,\n                                                                          tile_sizze_12709)) -\n                            squot32(squot32(global_tid_12478, tile_sizze_12709 *\n                                            tile_sizze_12709) -\n                                    squot32(squot32(global_tid_12478,\n                                                    tile_sizze_12709 *\n                                                    tile_sizze_12709),\n                                            (res_12055 + 1 - 1) * squot32(4 +\n                                                                          tile_sizze_12709 -\n                                                                          1,\n                                                                          tile_sizze_12709) *\n                                            squot32(4 + tile_sizze_12709 - 1,\n                                                    tile_sizze_12709)) *\n                                    ((res_12055 + 1 - 1) * squot32(4 +\n                                                                   tile_sizze_12709 -\n                                                                   1,\n                                                                   tile_sizze_12709) *\n                                     squot32(4 + tile_sizze_12709 - 1,\n                                             tile_sizze_12709)), squot32(4 +\n                                                                         tile_sizze_12709 -\n                                                                         1,\n                                                                         tile_sizze_12709) *\n                                    squot32(4 + tile_sizze_12709 - 1,\n                                            tile_sizze_12709)) * (squot32(4 +\n    ",
            "                                                                      tile_sizze_12709 -\n                                                                          1,\n                                                                          tile_sizze_12709) *\n                                                                  squot32(4 +\n                                                                          tile_sizze_12709 -\n                                                                          1,\n                                                                          tile_sizze_12709)) -\n                            squot32(squot32(global_tid_12478, tile_sizze_12709 *\n                                            tile_sizze_12709) -\n                                    squot32(squot32(global_tid_12478,\n                                                    tile_sizze_12709 *\n                                                    tile_sizze_12709),\n                                            (res_12055 + 1 - 1) * squot32(4 +\n                                                                          tile_sizze_12709 -\n                                                                          1,\n                                                                          tile_sizze_12709) *\n                                            squot32(4 + tile_sizze_12709 - 1,\n                                                    tile_sizze_12709)) *\n                                    ((res_12055 + 1 - 1) * squot32(4 +\n                                                                   tile_sizze_12709 -\n                                                                   1,\n                                                                   tile_sizze_12709) *\n                                     squot32(4 + tile_sizze_12709 - 1,\n                                             tile_sizze_12709)) -\n                                    squot32(squot32(global_tid_12478,\n                          ",
            "                          tile_sizze_12709 *\n                                                    tile_sizze_12709) -\n                                            squot32(squot32(global_tid_12478,\n                                                            tile_sizze_12709 *\n                                                            tile_sizze_12709),\n                                                    (res_12055 + 1 - 1) *\n                                                    squot32(4 +\n                                                            tile_sizze_12709 -\n                                                            1,\n                                                            tile_sizze_12709) *\n                                                    squot32(4 +\n                                                            tile_sizze_12709 -\n                                                            1,\n                                                            tile_sizze_12709)) *\n                                            ((res_12055 + 1 - 1) * squot32(4 +\n                                                                           tile_sizze_12709 -\n                                                                           1,\n                                                                           tile_sizze_12709) *\n                                             squot32(4 + tile_sizze_12709 - 1,\n                                                     tile_sizze_12709)),\n                                            squot32(4 + tile_sizze_12709 - 1,\n                                                    tile_sizze_12709) *\n                                            squot32(4 + tile_sizze_12709 - 1,\n                                                    tile_sizze_12709)) *\n                                    (squot32(4 + tile_sizze_12709 - 1,\n                                             tile_sizze_12709) * squot32(4 +\n                                                     ",
            "                    tile_sizze_12709 -\n                                                                         1,\n                                                                         tile_sizze_12709)),\n                                    squot32(4 + tile_sizze_12709 - 1,\n                                            tile_sizze_12709)) * squot32(4 +\n                                                                         tile_sizze_12709 -\n                                                                         1,\n                                                                         tile_sizze_12709)) *\n        tile_sizze_12709;\n    ltid_12711 = squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                tile_sizze_12709), tile_sizze_12709 *\n                         tile_sizze_12709);\n    ltid_12712 = squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                tile_sizze_12709) -\n                         squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                        tile_sizze_12709), tile_sizze_12709 *\n                                 tile_sizze_12709) * (tile_sizze_12709 *\n                                                      tile_sizze_12709),\n                         tile_sizze_12709 * tile_sizze_12709);\n    ltid_12713 = squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                tile_sizze_12709) -\n                         squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                        tile_sizze_12709), tile_sizze_12709 *\n                                 tile_sizze_12709) * (tile_sizze_12709 *\n                                                      tile_sizze_12709) -\n                         squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                        tile_sizze_12709) -\n                                 squot32(srem32(global_tid_12478,\n                                                tile_sizze_1270",
            "9 *\n                                                tile_sizze_12709),\n                                         tile_sizze_12709 * tile_sizze_12709) *\n                                 (tile_sizze_12709 * tile_sizze_12709),\n                                 tile_sizze_12709 * tile_sizze_12709) *\n                         (tile_sizze_12709 * tile_sizze_12709),\n                         tile_sizze_12709);\n    ltid_12714 = srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709) -\n        squot32(srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709),\n                tile_sizze_12709 * tile_sizze_12709) * (tile_sizze_12709 *\n                                                        tile_sizze_12709) -\n        squot32(srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709) -\n                squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                               tile_sizze_12709), tile_sizze_12709 *\n                        tile_sizze_12709) * (tile_sizze_12709 *\n                                             tile_sizze_12709),\n                tile_sizze_12709 * tile_sizze_12709) * (tile_sizze_12709 *\n                                                        tile_sizze_12709) -\n        squot32(srem32(global_tid_12478, tile_sizze_12709 * tile_sizze_12709) -\n                squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                               tile_sizze_12709), tile_sizze_12709 *\n                        tile_sizze_12709) * (tile_sizze_12709 *\n                                             tile_sizze_12709) -\n                squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                               tile_sizze_12709) -\n                        squot32(srem32(global_tid_12478, tile_sizze_12709 *\n                                       tile_sizze_12709), tile_sizze_12709 *\n                                tile_sizze_12709) * (tile_sizze_12709 *\n                                                     tile_sizze_12709),\n                        tile",
            "_sizze_12709 * tile_sizze_12709) *\n                (tile_sizze_12709 * tile_sizze_12709), tile_sizze_12709) *\n        tile_sizze_12709;\n    if (((slt32(gtid_12465, res_12052) && slt32(gtid_12466, res_12055)) &&\n         slt32(gtid_12467, 4)) && slt32(gtid_12468, 4)) { }\n    \n    float res_12483;\n    float x_12486 = 0.0F;\n    int32_t chunk_sizze_12484;\n    int32_t chunk_offset_12485 = 0;\n    \n    while (slt32(chunk_offset_12485, 4)) {\n        if (slt32(4 - chunk_offset_12485, tile_sizze_12709)) {\n            chunk_sizze_12484 = 4 - chunk_offset_12485;\n        } else {\n            chunk_sizze_12484 = tile_sizze_12709;\n        }\n        for (int32_t comb_iter_13042 = 0; comb_iter_13042 <\n             squot32(tile_sizze_12709 * tile_sizze_12709 +\n                     tiled_group_sizze_12710 - 1, tiled_group_sizze_12710);\n             comb_iter_13042++) {\n            int32_t cid_12734;\n            int32_t cid_12735;\n            int32_t flat_comb_id_13043 = comb_iter_13042 *\n                    tiled_group_sizze_12710 + local_tid_12479;\n            \n            cid_12734 = squot32(flat_comb_id_13043, tile_sizze_12709);\n            cid_12735 = flat_comb_id_13043 - squot32(flat_comb_id_13043,\n                                                     tile_sizze_12709) *\n                tile_sizze_12709;\n            if ((slt32(cid_12734, tile_sizze_12709) && slt32(cid_12735,\n                                                             chunk_sizze_12484)) &&\n                slt32(gtid_12467, 4)) {\n                float x_chunk_outer_elem_12733 = *(__global\n                                                   float *) &mem_12901[(gtid_12465 *\n                                                                        (16 *\n                                                                         res_12055) +\n                                                                        gtid_12466 *\n                                                                        16 +\n                      ",
            "                                                  gtid_12467 *\n                                                                        4 +\n                                                                        chunk_offset_12485 +\n                                                                        ltid_12714) *\n                                                                       4];\n                \n                *(__local float *) &mem_12905[(cid_12734 * tile_sizze_12709 +\n                                               cid_12735) * 4] =\n                    x_chunk_outer_elem_12733;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12465, res_12052) && slt32(gtid_12466, res_12055)) &&\n             slt32(gtid_12467, 4)) && slt32(gtid_12468, 4)) { }\n        for (int32_t comb_iter_13044 = 0; comb_iter_13044 <\n             squot32(tile_sizze_12709 * tile_sizze_12709 +\n                     tiled_group_sizze_12710 - 1, tiled_group_sizze_12710);\n             comb_iter_13044++) {\n            int32_t cid_12739;\n            int32_t cid_12740;\n            int32_t flat_comb_id_13045 = comb_iter_13044 *\n                    tiled_group_sizze_12710 + local_tid_12479;\n            \n            cid_12739 = squot32(flat_comb_id_13045, tile_sizze_12709);\n            cid_12740 = flat_comb_id_13045 - squot32(flat_comb_id_13045,\n                                                     tile_sizze_12709) *\n                tile_sizze_12709;\n            if ((slt32(cid_12739, chunk_sizze_12484) && slt32(cid_12740,\n                                                              tile_sizze_12709)) &&\n                slt32(gtid_12468, 4)) {\n                float x_chunk_outer_elem_12738 = *(__global\n                                                   float *) &mem_12875[(gtid_12468 +\n                                                                        4 *\n                                                                        chunk_offset_12485 +\n        ",
            "                                                                ltid_12713 *\n                                                                        4) * 4];\n                \n                *(__local float *) &mem_12909[(cid_12739 * tile_sizze_12709 +\n                                               cid_12740) * 4] =\n                    x_chunk_outer_elem_12738;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12465, res_12052) && slt32(gtid_12466, res_12055)) &&\n             slt32(gtid_12467, 4)) && slt32(gtid_12468, 4)) { }\n        \n        float res_12489;\n        float sync_12742;\n        float acc_12492 = x_12486;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_12490;\n        \n        groupstream_mapaccum_dummy_chunk_sizze_12490 = 1;\n        if (((slt32(gtid_12465, res_12052) && slt32(gtid_12466, res_12055)) &&\n             slt32(gtid_12467, 4)) && slt32(gtid_12468, 4)) {\n            if (chunk_sizze_12484 == tile_sizze_12709) {\n                for (int32_t i_12491 = 0; i_12491 < tile_sizze_12709;\n                     i_12491++) {\n                    float x_12495;\n                    float x_12496;\n                    float res_12498;\n                    float res_12500;\n                    \n                    x_12495 = *(__local float *) &mem_12905[(ltid_12713 *\n                                                             tile_sizze_12709 +\n                                                             i_12491) * 4];\n                    x_12496 = *(__local float *) &mem_12909[(tile_sizze_12709 *\n                                                             0 + ltid_12714 +\n                                                             tile_sizze_12709 *\n                                                             i_12491 + 0 *\n                                                             tile_sizze_12709) *\n                                                            4];\n                    res_12498 = x_12495 * x_12496;\n ",
            "                   res_12500 = acc_12492 + res_12498;\n                    \n                    float acc_tmp_13046 = res_12500;\n                    \n                    acc_12492 = acc_tmp_13046;\n                }\n            } else {\n                for (int32_t i_12491 = 0; i_12491 < chunk_sizze_12484;\n                     i_12491++) {\n                    float x_12495;\n                    float x_12496;\n                    float res_12498;\n                    float res_12500;\n                    \n                    x_12495 = *(__local float *) &mem_12905[(ltid_12713 *\n                                                             tile_sizze_12709 +\n                                                             i_12491) * 4];\n                    x_12496 = *(__local float *) &mem_12909[(tile_sizze_12709 *\n                                                             0 + ltid_12714 +\n                                                             tile_sizze_12709 *\n                                                             i_12491 + 0 *\n                                                             tile_sizze_12709) *\n                                                            4];\n                    res_12498 = x_12495 * x_12496;\n                    res_12500 = acc_12492 + res_12498;\n                    \n                    float acc_tmp_13047 = res_12500;\n                    \n                    acc_12492 = acc_tmp_13047;\n                }\n            }\n        }\n        res_12489 = acc_12492;\n        sync_12742 = res_12489;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_12486 = sync_12742;\n        chunk_offset_12485 += tile_sizze_12709;\n    }\n    res_12483 = x_12486;\n    if (((slt32(gtid_12465, res_12052) && slt32(gtid_12466, res_12055)) &&\n         slt32(gtid_12467, 4)) && slt32(gtid_12468, 4)) {\n        *(__global float *) &mem_12915[(gtid_12465 * (16 * res_12055) +\n                                        gtid_12466 * 16 + gtid_12467 * 4 +\n                           ",
            "             gtid_12468) * 4] = res_12483;\n    }\n}\n__kernel void map_12527(__local volatile int64_t *mem_aligned_0,\n                        __local volatile int64_t *mem_aligned_1,\n                        int32_t res_12052, int32_t res_12055, __global\n                        unsigned char *mem_12875, __global\n                        unsigned char *mem_12887, __global\n                        unsigned char *mem_12901)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_12891 = mem_aligned_0;\n    __local volatile char *restrict mem_12895 = mem_aligned_1;\n    int32_t wave_sizze_13032;\n    int32_t group_sizze_13033;\n    int32_t gtid_12514;\n    int32_t gtid_12515;\n    int32_t gtid_12516;\n    int32_t gtid_12517;\n    int32_t global_tid_12527;\n    int32_t local_tid_12528;\n    int32_t group_id_12529;\n    int32_t ltid_12676;\n    int32_t ltid_12677;\n    int32_t ltid_12678;\n    int32_t ltid_12679;\n    \n    global_tid_12527 = get_global_id(0);\n    local_tid_12528 = get_local_id(0);\n    group_sizze_13033 = get_local_size(0);\n    wave_sizze_13032 = LOCKSTEP_WIDTH;\n    group_id_12529 = get_group_id(0);\n    gtid_12514 = squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                tile_sizze_12674), tile_sizze_12674 *\n                         tile_sizze_12674) + squot32(squot32(global_tid_12527,\n                                                             tile_sizze_12674 *\n                                                             tile_sizze_12674),\n                                                     (res_12055 + 1 - 1) *\n                                                     squot32(4 +\n                                                             tile_sizze_12674 -\n                                                             1,\n                                                             tile_sizze_12674) *\n                                                     squot32(4 +\n         ",
            "                                                    tile_sizze_12674 -\n                                                             1,\n                                                             tile_sizze_12674));\n    gtid_12515 = squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                tile_sizze_12674) -\n                         squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                        tile_sizze_12674), tile_sizze_12674 *\n                                 tile_sizze_12674) * (tile_sizze_12674 *\n                                                      tile_sizze_12674),\n                         tile_sizze_12674 * tile_sizze_12674) +\n        squot32(squot32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674) -\n                squot32(squot32(global_tid_12527, tile_sizze_12674 *\n                                tile_sizze_12674), (res_12055 + 1 - 1) *\n                        squot32(4 + tile_sizze_12674 - 1, tile_sizze_12674) *\n                        squot32(4 + tile_sizze_12674 - 1, tile_sizze_12674)) *\n                ((res_12055 + 1 - 1) * squot32(4 + tile_sizze_12674 - 1,\n                                               tile_sizze_12674) * squot32(4 +\n                                                                           tile_sizze_12674 -\n                                                                           1,\n                                                                           tile_sizze_12674)),\n                squot32(4 + tile_sizze_12674 - 1, tile_sizze_12674) *\n                squot32(4 + tile_sizze_12674 - 1, tile_sizze_12674));\n    gtid_12516 = squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                tile_sizze_12674) -\n                         squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                        tile_sizze_12674), tile_sizze_12674 *\n                                 tile_sizze_12674) * (tile_sizze_12674 *\n          ",
            "                                            tile_sizze_12674) -\n                         squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                        tile_sizze_12674) -\n                                 squot32(srem32(global_tid_12527,\n                                                tile_sizze_12674 *\n                                                tile_sizze_12674),\n                                         tile_sizze_12674 * tile_sizze_12674) *\n                                 (tile_sizze_12674 * tile_sizze_12674),\n                                 tile_sizze_12674 * tile_sizze_12674) *\n                         (tile_sizze_12674 * tile_sizze_12674),\n                         tile_sizze_12674) + squot32(squot32(global_tid_12527,\n                                                             tile_sizze_12674 *\n                                                             tile_sizze_12674) -\n                                                     squot32(squot32(global_tid_12527,\n                                                                     tile_sizze_12674 *\n                                                                     tile_sizze_12674),\n                                                             (res_12055 + 1 -\n                                                              1) * squot32(4 +\n                                                                           tile_sizze_12674 -\n                                                                           1,\n                                                                           tile_sizze_12674) *\n                                                             squot32(4 +\n                                                                     tile_sizze_12674 -\n                                                                     1,\n                                                                     tile_sizze_12674)) *\n                                                     ((res_12",
            "055 + 1 - 1) *\n                                                      squot32(4 +\n                                                              tile_sizze_12674 -\n                                                              1,\n                                                              tile_sizze_12674) *\n                                                      squot32(4 +\n                                                              tile_sizze_12674 -\n                                                              1,\n                                                              tile_sizze_12674)) -\n                                                     squot32(squot32(global_tid_12527,\n                                                                     tile_sizze_12674 *\n                                                                     tile_sizze_12674) -\n                                                             squot32(squot32(global_tid_12527,\n                                                                             tile_sizze_12674 *\n                                                                             tile_sizze_12674),\n                                                                     (res_12055 +\n                                                                      1 - 1) *\n                                                                     squot32(4 +\n                                                                             tile_sizze_12674 -\n                                                                             1,\n                                                                             tile_sizze_12674) *\n                                                                     squot32(4 +\n                                                                             tile_sizze_12674 -\n                                                                             1,\n                                                                         ",
            "    tile_sizze_12674)) *\n                                                             ((res_12055 + 1 -\n                                                               1) * squot32(4 +\n                                                                            tile_sizze_12674 -\n                                                                            1,\n                                                                            tile_sizze_12674) *\n                                                              squot32(4 +\n                                                                      tile_sizze_12674 -\n                                                                      1,\n                                                                      tile_sizze_12674)),\n                                                             squot32(4 +\n                                                                     tile_sizze_12674 -\n                                                                     1,\n                                                                     tile_sizze_12674) *\n                                                             squot32(4 +\n                                                                     tile_sizze_12674 -\n                                                                     1,\n                                                                     tile_sizze_12674)) *\n                                                     (squot32(4 +\n                                                              tile_sizze_12674 -\n                                                              1,\n                                                              tile_sizze_12674) *\n                                                      squot32(4 +\n                                                              tile_sizze_12674 -\n                                                              1,\n                                                              tile_si",
            "zze_12674)),\n                                                     squot32(4 +\n                                                             tile_sizze_12674 -\n                                                             1,\n                                                             tile_sizze_12674)) *\n        tile_sizze_12674;\n    gtid_12517 = srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674) -\n        squot32(srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674),\n                tile_sizze_12674 * tile_sizze_12674) * (tile_sizze_12674 *\n                                                        tile_sizze_12674) -\n        squot32(srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674) -\n                squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                               tile_sizze_12674), tile_sizze_12674 *\n                        tile_sizze_12674) * (tile_sizze_12674 *\n                                             tile_sizze_12674),\n                tile_sizze_12674 * tile_sizze_12674) * (tile_sizze_12674 *\n                                                        tile_sizze_12674) -\n        squot32(srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674) -\n                squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                               tile_sizze_12674), tile_sizze_12674 *\n                        tile_sizze_12674) * (tile_sizze_12674 *\n                                             tile_sizze_12674) -\n                squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                               tile_sizze_12674) -\n                        squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                       tile_sizze_12674), tile_sizze_12674 *\n                                tile_sizze_12674) * (tile_sizze_12674 *\n                                                     tile_sizze_12674),\n                        tile_sizze_12674 * tile_sizze_12674) *\n                (tile_sizze_12674 * ti",
            "le_sizze_12674), tile_sizze_12674) *\n        tile_sizze_12674 + (squot32(global_tid_12527, tile_sizze_12674 *\n                                    tile_sizze_12674) -\n                            squot32(squot32(global_tid_12527, tile_sizze_12674 *\n                                            tile_sizze_12674), (res_12055 + 1 -\n                                                                1) * squot32(4 +\n                                                                             tile_sizze_12674 -\n                                                                             1,\n                                                                             tile_sizze_12674) *\n                                    squot32(4 + tile_sizze_12674 - 1,\n                                            tile_sizze_12674)) * ((res_12055 +\n                                                                   1 - 1) *\n                                                                  squot32(4 +\n                                                                          tile_sizze_12674 -\n                                                                          1,\n                                                                          tile_sizze_12674) *\n                                                                  squot32(4 +\n                                                                          tile_sizze_12674 -\n                                                                          1,\n                                                                          tile_sizze_12674)) -\n                            squot32(squot32(global_tid_12527, tile_sizze_12674 *\n                                            tile_sizze_12674) -\n                                    squot32(squot32(global_tid_12527,\n                                                    tile_sizze_12674 *\n                                                    tile_sizze_12674),\n                                            (res_120",
            "55 + 1 - 1) * squot32(4 +\n                                                                          tile_sizze_12674 -\n                                                                          1,\n                                                                          tile_sizze_12674) *\n                                            squot32(4 + tile_sizze_12674 - 1,\n                                                    tile_sizze_12674)) *\n                                    ((res_12055 + 1 - 1) * squot32(4 +\n                                                                   tile_sizze_12674 -\n                                                                   1,\n                                                                   tile_sizze_12674) *\n                                     squot32(4 + tile_sizze_12674 - 1,\n                                             tile_sizze_12674)), squot32(4 +\n                                                                         tile_sizze_12674 -\n                                                                         1,\n                                                                         tile_sizze_12674) *\n                                    squot32(4 + tile_sizze_12674 - 1,\n                                            tile_sizze_12674)) * (squot32(4 +\n                                                                          tile_sizze_12674 -\n                                                                          1,\n                                                                          tile_sizze_12674) *\n                                                                  squot32(4 +\n                                                                          tile_sizze_12674 -\n                                                                          1,\n                                                                          tile_sizze_12674)) -\n                            squot32(squot32(global_tid_12527, tile_sizze_12674 *\n",
            "                                            tile_sizze_12674) -\n                                    squot32(squot32(global_tid_12527,\n                                                    tile_sizze_12674 *\n                                                    tile_sizze_12674),\n                                            (res_12055 + 1 - 1) * squot32(4 +\n                                                                          tile_sizze_12674 -\n                                                                          1,\n                                                                          tile_sizze_12674) *\n                                            squot32(4 + tile_sizze_12674 - 1,\n                                                    tile_sizze_12674)) *\n                                    ((res_12055 + 1 - 1) * squot32(4 +\n                                                                   tile_sizze_12674 -\n                                                                   1,\n                                                                   tile_sizze_12674) *\n                                     squot32(4 + tile_sizze_12674 - 1,\n                                             tile_sizze_12674)) -\n                                    squot32(squot32(global_tid_12527,\n                                                    tile_sizze_12674 *\n                                                    tile_sizze_12674) -\n                                            squot32(squot32(global_tid_12527,\n                                                            tile_sizze_12674 *\n                                                            tile_sizze_12674),\n                                                    (res_12055 + 1 - 1) *\n                                                    squot32(4 +\n                                                            tile_sizze_12674 -\n                                                            1,\n                                                   ",
            "         tile_sizze_12674) *\n                                                    squot32(4 +\n                                                            tile_sizze_12674 -\n                                                            1,\n                                                            tile_sizze_12674)) *\n                                            ((res_12055 + 1 - 1) * squot32(4 +\n                                                                           tile_sizze_12674 -\n                                                                           1,\n                                                                           tile_sizze_12674) *\n                                             squot32(4 + tile_sizze_12674 - 1,\n                                                     tile_sizze_12674)),\n                                            squot32(4 + tile_sizze_12674 - 1,\n                                                    tile_sizze_12674) *\n                                            squot32(4 + tile_sizze_12674 - 1,\n                                                    tile_sizze_12674)) *\n                                    (squot32(4 + tile_sizze_12674 - 1,\n                                             tile_sizze_12674) * squot32(4 +\n                                                                         tile_sizze_12674 -\n                                                                         1,\n                                                                         tile_sizze_12674)),\n                                    squot32(4 + tile_sizze_12674 - 1,\n                                            tile_sizze_12674)) * squot32(4 +\n                                                                         tile_sizze_12674 -\n                                                                         1,\n                                                                         tile_sizze_12674)) *\n        tile_sizze_12674;\n    ltid_12676 = squot32(srem32(global_ti",
            "d_12527, tile_sizze_12674 *\n                                tile_sizze_12674), tile_sizze_12674 *\n                         tile_sizze_12674);\n    ltid_12677 = squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                tile_sizze_12674) -\n                         squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                        tile_sizze_12674), tile_sizze_12674 *\n                                 tile_sizze_12674) * (tile_sizze_12674 *\n                                                      tile_sizze_12674),\n                         tile_sizze_12674 * tile_sizze_12674);\n    ltid_12678 = squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                tile_sizze_12674) -\n                         squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                        tile_sizze_12674), tile_sizze_12674 *\n                                 tile_sizze_12674) * (tile_sizze_12674 *\n                                                      tile_sizze_12674) -\n                         squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                        tile_sizze_12674) -\n                                 squot32(srem32(global_tid_12527,\n                                                tile_sizze_12674 *\n                                                tile_sizze_12674),\n                                         tile_sizze_12674 * tile_sizze_12674) *\n                                 (tile_sizze_12674 * tile_sizze_12674),\n                                 tile_sizze_12674 * tile_sizze_12674) *\n                         (tile_sizze_12674 * tile_sizze_12674),\n                         tile_sizze_12674);\n    ltid_12679 = srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674) -\n        squot32(srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674),\n                tile_sizze_12674 * tile_sizze_12674) * (tile_sizze_12674 *\n                                              ",
            "          tile_sizze_12674) -\n        squot32(srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674) -\n                squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                               tile_sizze_12674), tile_sizze_12674 *\n                        tile_sizze_12674) * (tile_sizze_12674 *\n                                             tile_sizze_12674),\n                tile_sizze_12674 * tile_sizze_12674) * (tile_sizze_12674 *\n                                                        tile_sizze_12674) -\n        squot32(srem32(global_tid_12527, tile_sizze_12674 * tile_sizze_12674) -\n                squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                               tile_sizze_12674), tile_sizze_12674 *\n                        tile_sizze_12674) * (tile_sizze_12674 *\n                                             tile_sizze_12674) -\n                squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                               tile_sizze_12674) -\n                        squot32(srem32(global_tid_12527, tile_sizze_12674 *\n                                       tile_sizze_12674), tile_sizze_12674 *\n                                tile_sizze_12674) * (tile_sizze_12674 *\n                                                     tile_sizze_12674),\n                        tile_sizze_12674 * tile_sizze_12674) *\n                (tile_sizze_12674 * tile_sizze_12674), tile_sizze_12674) *\n        tile_sizze_12674;\n    if (((slt32(gtid_12514, res_12052) && slt32(gtid_12515, res_12055)) &&\n         slt32(gtid_12516, 4)) && slt32(gtid_12517, 4)) { }\n    \n    float res_12532;\n    float x_12535 = 0.0F;\n    int32_t chunk_sizze_12533;\n    int32_t chunk_offset_12534 = 0;\n    \n    while (slt32(chunk_offset_12534, 4)) {\n        if (slt32(4 - chunk_offset_12534, tile_sizze_12674)) {\n            chunk_sizze_12533 = 4 - chunk_offset_12534;\n        } else {\n            chunk_sizze_12533 = tile_sizze_12674;\n        }\n        for (int32_t comb_iter_13034 = 0; comb_ite",
            "r_13034 <\n             squot32(tile_sizze_12674 * tile_sizze_12674 +\n                     tiled_group_sizze_12675 - 1, tiled_group_sizze_12675);\n             comb_iter_13034++) {\n            int32_t cid_12699;\n            int32_t cid_12700;\n            int32_t flat_comb_id_13035 = comb_iter_13034 *\n                    tiled_group_sizze_12675 + local_tid_12528;\n            \n            cid_12699 = squot32(flat_comb_id_13035, tile_sizze_12674);\n            cid_12700 = flat_comb_id_13035 - squot32(flat_comb_id_13035,\n                                                     tile_sizze_12674) *\n                tile_sizze_12674;\n            if ((slt32(cid_12699, tile_sizze_12674) && slt32(cid_12700,\n                                                             chunk_sizze_12533)) &&\n                slt32(gtid_12516, 4)) {\n                float x_chunk_outer_elem_12698 = *(__global\n                                                   float *) &mem_12875[(gtid_12516 +\n                                                                        4 *\n                                                                        chunk_offset_12534 +\n                                                                        ltid_12679 *\n                                                                        4) * 4];\n                \n                *(__local float *) &mem_12891[(cid_12699 * tile_sizze_12674 +\n                                               cid_12700) * 4] =\n                    x_chunk_outer_elem_12698;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12514, res_12052) && slt32(gtid_12515, res_12055)) &&\n             slt32(gtid_12516, 4)) && slt32(gtid_12517, 4)) { }\n        for (int32_t comb_iter_13036 = 0; comb_iter_13036 <\n             squot32(tile_sizze_12674 * tile_sizze_12674 +\n                     tiled_group_sizze_12675 - 1, tiled_group_sizze_12675);\n             comb_iter_13036++) {\n            int32_t cid_12704;\n            int32_t cid_12",
            "705;\n            int32_t flat_comb_id_13037 = comb_iter_13036 *\n                    tiled_group_sizze_12675 + local_tid_12528;\n            \n            cid_12704 = squot32(flat_comb_id_13037, tile_sizze_12674);\n            cid_12705 = flat_comb_id_13037 - squot32(flat_comb_id_13037,\n                                                     tile_sizze_12674) *\n                tile_sizze_12674;\n            if ((slt32(cid_12704, chunk_sizze_12533) && slt32(cid_12705,\n                                                              tile_sizze_12674)) &&\n                slt32(gtid_12517, 4)) {\n                float x_chunk_outer_elem_12703 = *(__global\n                                                   float *) &mem_12887[(gtid_12514 *\n                                                                        (4 *\n                                                                         res_12055) +\n                                                                        gtid_12515 *\n                                                                        4 +\n                                                                        gtid_12517 +\n                                                                        4 *\n                                                                        res_12055 *\n                                                                        res_12052 *\n                                                                        chunk_offset_12534 +\n                                                                        ltid_12678 *\n                                                                        (4 *\n                                                                         res_12055 *\n                                                                         res_12052)) *\n                                                                       4];\n                \n                *(__local float *) &mem_12895[(cid_12704 * tile_sizze_12674 +\n                 ",
            "                              cid_12705) * 4] =\n                    x_chunk_outer_elem_12703;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (((slt32(gtid_12514, res_12052) && slt32(gtid_12515, res_12055)) &&\n             slt32(gtid_12516, 4)) && slt32(gtid_12517, 4)) { }\n        \n        float res_12538;\n        float sync_12707;\n        float acc_12541 = x_12535;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_12539;\n        \n        groupstream_mapaccum_dummy_chunk_sizze_12539 = 1;\n        if (((slt32(gtid_12514, res_12052) && slt32(gtid_12515, res_12055)) &&\n             slt32(gtid_12516, 4)) && slt32(gtid_12517, 4)) {\n            if (chunk_sizze_12533 == tile_sizze_12674) {\n                for (int32_t i_12540 = 0; i_12540 < tile_sizze_12674;\n                     i_12540++) {\n                    float x_12544;\n                    float x_12545;\n                    float res_12547;\n                    float res_12549;\n                    \n                    x_12544 = *(__local float *) &mem_12891[(ltid_12678 *\n                                                             tile_sizze_12674 +\n                                                             i_12540) * 4];\n                    x_12545 = *(__local float *) &mem_12895[(tile_sizze_12674 *\n                                                             0 + ltid_12679 +\n                                                             tile_sizze_12674 *\n                                                             i_12540 + 0 *\n                                                             tile_sizze_12674) *\n                                                            4];\n                    res_12547 = x_12544 * x_12545;\n                    res_12549 = acc_12541 + res_12547;\n                    \n                    float acc_tmp_13038 = res_12549;\n                    \n                    acc_12541 = acc_tmp_13038;\n                }\n            } else {\n                for (int32_t i_1254",
            "0 = 0; i_12540 < chunk_sizze_12533;\n                     i_12540++) {\n                    float x_12544;\n                    float x_12545;\n                    float res_12547;\n                    float res_12549;\n                    \n                    x_12544 = *(__local float *) &mem_12891[(ltid_12678 *\n                                                             tile_sizze_12674 +\n                                                             i_12540) * 4];\n                    x_12545 = *(__local float *) &mem_12895[(tile_sizze_12674 *\n                                                             0 + ltid_12679 +\n                                                             tile_sizze_12674 *\n                                                             i_12540 + 0 *\n                                                             tile_sizze_12674) *\n                                                            4];\n                    res_12547 = x_12544 * x_12545;\n                    res_12549 = acc_12541 + res_12547;\n                    \n                    float acc_tmp_13039 = res_12549;\n                    \n                    acc_12541 = acc_tmp_13039;\n                }\n            }\n        }\n        res_12538 = acc_12541;\n        sync_12707 = res_12538;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_12535 = sync_12707;\n        chunk_offset_12534 += tile_sizze_12674;\n    }\n    res_12532 = x_12535;\n    if (((slt32(gtid_12514, res_12052) && slt32(gtid_12515, res_12055)) &&\n         slt32(gtid_12516, 4)) && slt32(gtid_12517, 4)) {\n        *(__global float *) &mem_12901[(gtid_12514 * (16 * res_12055) +\n                                        gtid_12515 * 16 + gtid_12516 * 4 +\n                                        gtid_12517) * 4] = res_12532;\n    }\n}\n__kernel void map_12562(int32_t num_elems_11987, int32_t res_12052,\n                        int32_t res_12055, int32_t num_threads_12561,\n                        int32_t per_chunk_12603, __global\n                       ",
            " unsigned char *mem_12867, __global\n                        unsigned char *mem_12872)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t wave_sizze_13028;\n    int32_t group_sizze_13029;\n    int32_t gtid_12553;\n    int32_t gtid_12554;\n    int32_t global_tid_12562;\n    int32_t local_tid_12563;\n    int32_t group_id_12564;\n    \n    global_tid_12562 = get_global_id(0);\n    local_tid_12563 = get_local_id(0);\n    group_sizze_13029 = get_local_size(0);\n    wave_sizze_13028 = LOCKSTEP_WIDTH;\n    group_id_12564 = get_group_id(0);\n    gtid_12553 = squot32(global_tid_12562, res_12055);\n    gtid_12554 = global_tid_12562 - squot32(global_tid_12562, res_12055) *\n        res_12055;\n    \n    int32_t index_primexp_12596;\n    int32_t index_primexp_12595;\n    \n    if (slt32(gtid_12553, res_12052) && slt32(gtid_12554, res_12055)) {\n        index_primexp_12596 = 2 * gtid_12553;\n        index_primexp_12595 = 2 * gtid_12554;\n    }\n    if (slt32(gtid_12553, res_12052) && slt32(gtid_12554, res_12055)) {\n        for (int32_t i_13030 = 0; i_13030 < 4; i_13030++) {\n            for (int32_t i_13031 = 0; i_13031 < 4; i_13031++) {\n                *(__global float *) &mem_12872[(gtid_12553 * res_12055 +\n                                                gtid_12554 + (i_13030 *\n                                                              (res_12055 *\n                                                               res_12052 * 4) +\n                                                              i_13031 *\n                                                              (res_12055 *\n                                                               res_12052))) *\n                                               4] = *(__global\n                                                      float *) &mem_12867[(squot32(num_elems_11987 *\n                                                                                   index_primexp_12596 +\n                                    ",
            "                                               index_primexp_12595 +\n                                                                                   (i_13031 *\n                                                                                    num_elems_11987 +\n                                                                                    i_13030) -\n                                                                                   squot32(num_elems_11987 *\n                                                                                           index_primexp_12596 +\n                                                                                           index_primexp_12595 +\n                                                                                           (i_13031 *\n                                                                                            num_elems_11987 +\n                                                                                            i_13030),\n                                                                                           per_chunk_12603 *\n                                                                                           num_elems_11987) *\n                                                                                   (per_chunk_12603 *\n                                                                                    num_elems_11987),\n                                                                                   num_elems_11987) *\n                                                                           (num_threads_12561 *\n                                                                            num_elems_11987) +\n                                                                           (num_elems_11987 *\n                                                                            index_primexp_12596 +\n                                                                            index_p",
            "rimexp_12595 +\n                                                                            (i_13031 *\n                                                                             num_elems_11987 +\n                                                                             i_13030) -\n                                                                            squot32(num_elems_11987 *\n                                                                                    index_primexp_12596 +\n                                                                                    index_primexp_12595 +\n                                                                                    (i_13031 *\n                                                                                     num_elems_11987 +\n                                                                                     i_13030),\n                                                                                    per_chunk_12603 *\n                                                                                    num_elems_11987) *\n                                                                            (per_chunk_12603 *\n                                                                             num_elems_11987) -\n                                                                            squot32(num_elems_11987 *\n                                                                                    index_primexp_12596 +\n                                                                                    index_primexp_12595 +\n                                                                                    (i_13031 *\n                                                                                     num_elems_11987 +\n                                                                                     i_13030) -\n                                                                                    squot32(num_elems_11",
            "987 *\n                                                                                            index_primexp_12596 +\n                                                                                            index_primexp_12595 +\n                                                                                            (i_13031 *\n                                                                                             num_elems_11987 +\n                                                                                             i_13030),\n                                                                                            per_chunk_12603 *\n                                                                                            num_elems_11987) *\n                                                                                    (per_chunk_12603 *\n                                                                                     num_elems_11987),\n                                                                                    num_elems_11987) *\n                                                                            num_elems_11987) *\n                                                                           num_threads_12561 +\n                                                                           squot32(num_elems_11987 *\n                                                                                   index_primexp_12596 +\n                                                                                   index_primexp_12595 +\n                                                                                   (i_13031 *\n                                                                                    num_elems_11987 +\n                                                                                    i_13030),\n                                                                                   per_chunk_12603 *\n                           ",
            "                                                        num_elems_11987)) *\n                                                                          4];\n            }\n        }\n    }\n}\n__kernel void map_transpose_f32(int32_t destoffset_1, int32_t srcoffset_3,\n                                int32_t num_arrays_4, int32_t x_elems_5,\n                                int32_t y_elems_6, int32_t in_elems_7,\n                                int32_t out_elems_8, int32_t mulx_9,\n                                int32_t muly_10, __global\n                                unsigned char *destmem_0, __global\n                                unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    ALIGNED_LOCAL_MEMORY(block_11, 4224);\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_global_id_0_37;\n    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;\n    \n    if (slt32(x_index_31, x_elems_5)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,\n                                                                 in_",
            "elems_7)) {\n                *(__local float *) &block_11[((get_local_id_1_39 + j_43 * 8) *\n                                              33 + get_local_id_0_38) *\n                                             sizeof(float)] = *(__global\n                                                                float *) &srcmem_2[(idata_offset_34 +\n                                                                                    index_in_35) *\n                                                                                   sizeof(float)];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;\n    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;\n    if (slt32(x_index_31, y_elems_6)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,\n                                                                 out_elems_8)) {\n                *(__global float *) &destmem_0[(odata_offset_33 +\n                                                index_out_36) * sizeof(float)] =\n                    *(__local float *) &block_11[(get_local_id_0_38 * 33 +\n                                                  get_local_id_1_39 + j_43 *\n                                                  8) * sizeof(float)];\n            }\n        }\n    }\n}\n__kernel void map_transpose_f32_low_height(int32_t destoffset_1,\n                                           int32_t srcoffset_3,\n                                           int32_t num_arrays_4,\n                                           int32_t x_elems_5, int32_t y_elems_6,\n                                           int32_t in_elems_7,\n                                           int32_t out_elems_8, int32_t mulx_9,\n                                           int32_t muly_10, __global\n                    ",
            "                       unsigned char *destmem_0, __global\n                                           unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    ALIGNED_LOCAL_MEMORY(block_11, 1088);\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +\n            srem32(get_local_id_1_39, mulx_9) * 16;\n    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,\n                                                          mulx_9);\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        *(__local float *) &block_11[(get_local_id_1_39 * 17 +\n                                      get_local_id_0_38) * sizeof(float)] =\n            *(__global float *) &srcmem_2[(idata_offset_34 + index_in_35) *\n                                          sizeof(float)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);\n    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +\n        srem32(get_local_id_0_38, mulx_9) * 16;\n ",
            "   \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *\n                                       sizeof(float)] = *(__local\n                                                          float *) &block_11[(get_local_id_0_38 *\n                                                                              17 +\n                                                                              get_local_id_1_39) *\n                                                                             sizeof(float)];\n    }\n}\n__kernel void map_transpose_f32_low_width(int32_t destoffset_1,\n                                          int32_t srcoffset_3,\n                                          int32_t num_arrays_4,\n                                          int32_t x_elems_5, int32_t y_elems_6,\n                                          int32_t in_elems_7,\n                                          int32_t out_elems_8, int32_t mulx_9,\n                                          int32_t muly_10, __global\n                                          unsigned char *destmem_0, __global\n                                          unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    ALIGNED_LOCAL_MEMORY(block_11, 1088);\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2",
            ");\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,\n                                                          muly_10);\n    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +\n            srem32(get_local_id_0_38, muly_10) * 16;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        *(__local float *) &block_11[(get_local_id_1_39 * 17 +\n                                      get_local_id_0_38) * sizeof(float)] =\n            *(__global float *) &srcmem_2[(idata_offset_34 + index_in_35) *\n                                          sizeof(float)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 * muly_10 + get_local_id_0_38 +\n        srem32(get_local_id_1_39, muly_10) * 16;\n    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *\n                                       sizeof(float)] = *(__local\n                                                          float *) &block_11[(get_local_id_0_38 *\n                                                                              17 +\n                                                                              get_local_id_1_39) *\n                                                                             sizeof(float)];\n    }\n}\n__kern",
            "el void map_transpose_f32_small(int32_t destoffset_1, int32_t srcoffset_3,\n                                      int32_t num_arrays_4, int32_t x_elems_5,\n                                      int32_t y_elems_6, int32_t in_elems_7,\n                                      int32_t out_elems_8, int32_t mulx_9,\n                                      int32_t muly_10, __global\n                                      unsigned char *destmem_0, __global\n                                      unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    ALIGNED_LOCAL_MEMORY(block_11, 1);\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *\n                                          x_elems_5) * (y_elems_6 * x_elems_5);\n    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *\n                                        x_elems_5), y_elems_6);\n    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;\n    \n    if (slt32(get_global_id_0_37, in_elems_7)) {\n        *(__global float *) &destmem_0[(odata_offset_33 + index_out_36) *\n                                       sizeof(float)] = *(__global\n                           ",
            "                               float *) &srcmem_2[(idata_offset_34 +\n                                                                              index_in_35) *\n                                                                             sizeof(float)];\n    }\n}\n__kernel void replicate_13075(int32_t sizze_12064, int32_t sizze_12065, __global\n                              unsigned char *mem_12989)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_13075;\n    int32_t replicate_ltid_13076;\n    int32_t replicate_gid_13077;\n    \n    replicate_gtid_13075 = get_global_id(0);\n    replicate_ltid_13076 = get_local_id(0);\n    replicate_gid_13077 = get_group_id(0);\n    if (slt32(replicate_gtid_13075, sizze_12064 * sizze_12065)) {\n        *(__global float *) &mem_12989[(squot32(replicate_gtid_13075,\n                                                sizze_12065) * sizze_12065 +\n                                        (replicate_gtid_13075 -\n                                         squot32(replicate_gtid_13075,\n                                                 sizze_12065) * sizze_12065)) *\n                                       4] = 2.0F;\n    }\n}\n",
            NULL};
static float static_array_realtype_13089[12] = {1.0F, 0.0F, 0.0F, 0.5F, 0.5F,
                                                0.5F, 0.5F, -0.5F, 0.5F, 0.0F,
                                                0.0F, 1.0F};
static float static_array_realtype_13101[16] = {1.0F, 0.0F, -1.0F, 0.0F, 0.0F,
                                                1.0F, 1.0F, 0.0F, 0.0F, -1.0F,
                                                1.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                                                -1.0F};
static float static_array_realtype_13102[8] = {1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
                                               1.0F, -1.0F, -1.0F};
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
    const char *desc;
} ;
struct memblock_local {
    int *references;
    unsigned char mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
static const char *size_names[] = {"main.group_size_12174",
                                   "main.group_size_12416",
                                   "main.group_size_12436",
                                   "main.group_size_12556",
                                   "main.group_size_13073",
                                   "main.group_size_13078",
                                   "main.tile_size_12623",
                                   "main.tile_size_12648",
                                   "main.tile_size_12673",
                                   "main.tile_size_12708",
                                   "main.tile_size_12743",
                                   "main.tile_size_12778"};
static const char *size_vars[] = {"mainzigroup_sizze_12174",
                                  "mainzigroup_sizze_12416",
                                  "mainzigroup_sizze_12436",
                                  "mainzigroup_sizze_12556",
                                  "mainzigroup_sizze_13073",
                                  "mainzigroup_sizze_13078",
                                  "mainzitile_sizze_12623",
                                  "mainzitile_sizze_12648",
                                  "mainzitile_sizze_12673",
                                  "mainzitile_sizze_12708",
                                  "mainzitile_sizze_12743",
                                  "mainzitile_sizze_12778"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "tile_size", "tile_size", "tile_size",
                                     "tile_size", "tile_size", "tile_size"};
int futhark_get_num_sizes(void)
{
    return 12;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    size_t mainzigroup_sizze_12174;
    size_t mainzigroup_sizze_12416;
    size_t mainzigroup_sizze_12436;
    size_t mainzigroup_sizze_12556;
    size_t mainzigroup_sizze_13073;
    size_t mainzigroup_sizze_13078;
    size_t mainzitile_sizze_12623;
    size_t mainzitile_sizze_12648;
    size_t mainzitile_sizze_12673;
    size_t mainzitile_sizze_12708;
    size_t mainzitile_sizze_12743;
    size_t mainzitile_sizze_12778;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[12];
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cfg->sizes[5] = 0;
    cfg->sizes[6] = 0;
    cfg->sizes[7] = 0;
    cfg->sizes[8] = 0;
    cfg->sizes[9] = 0;
    cfg->sizes[10] = 0;
    cfg->sizes[11] = 0;
    opencl_config_init(&cfg->opencl, 12, size_names, size_vars, cfg->sizes,
                       size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.logging = cfg->opencl.debugging = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->opencl.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->opencl, s);
}
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s)
{
    set_preferred_platform(&cfg->opencl, s);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->opencl.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->opencl.load_program_from = path;
}
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path)
{
    cfg->opencl.dump_binary_to = path;
}
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path)
{
    cfg->opencl.load_binary_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->opencl.default_group_size = size;
    cfg->opencl.default_group_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->opencl.default_num_groups = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_tile_size = size;
    cfg->opencl.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 12; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int logging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_local;
    int64_t cur_mem_usage_local;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct memblock_device static_array_13008;
    struct memblock_device static_array_13025;
    struct memblock_device static_array_13026;
    int total_runs;
    long total_runtime;
    cl_kernel copy_13070;
    int copy_13070_total_runtime;
    int copy_13070_runs;
    cl_kernel map_12180;
    int map_12180_total_runtime;
    int map_12180_runs;
    cl_kernel map_12229;
    int map_12229_total_runtime;
    int map_12229_runs;
    cl_kernel map_12272;
    int map_12272_total_runtime;
    int map_12272_runs;
    cl_kernel map_12337;
    int map_12337_total_runtime;
    int map_12337_runs;
    cl_kernel map_12386;
    int map_12386_total_runtime;
    int map_12386_runs;
    cl_kernel map_12422;
    int map_12422_total_runtime;
    int map_12422_runs;
    cl_kernel map_12442;
    int map_12442_total_runtime;
    int map_12442_runs;
    cl_kernel map_12478;
    int map_12478_total_runtime;
    int map_12478_runs;
    cl_kernel map_12527;
    int map_12527_total_runtime;
    int map_12527_runs;
    cl_kernel map_12562;
    int map_12562_total_runtime;
    int map_12562_runs;
    cl_kernel map_transpose_f32;
    int map_transpose_f32_total_runtime;
    int map_transpose_f32_runs;
    cl_kernel map_transpose_f32_low_height;
    int map_transpose_f32_low_height_total_runtime;
    int map_transpose_f32_low_height_runs;
    cl_kernel map_transpose_f32_low_width;
    int map_transpose_f32_low_width_total_runtime;
    int map_transpose_f32_low_width_runs;
    cl_kernel map_transpose_f32_small;
    int map_transpose_f32_small_total_runtime;
    int map_transpose_f32_small_runs;
    cl_kernel replicate_13075;
    int replicate_13075_total_runtime;
    int replicate_13075_runs;
    struct opencl_context opencl;
    struct sizes sizes;
} ;
void post_opencl_setup(struct opencl_context *ctx,
                       struct opencl_device_option *option)
{
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "NVIDIA CUDA") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "AMD Accelerated Parallel Processing") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 64;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_num_groups = 256;
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_group_size = 256;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_tile_size = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU)
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ctx->cfg.default_num_groups),
                        &ctx->cfg.default_num_groups, NULL);
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_group_size = 32;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_tile_size = 4;
}
static void init_context_early(struct futhark_context_config *cfg,
                               struct futhark_context *ctx)
{
    cl_int error;
    
    ctx->opencl.cfg = cfg->opencl;
    ctx->detail_memory = cfg->opencl.debugging;
    ctx->debugging = cfg->opencl.debugging;
    ctx->logging = cfg->opencl.logging;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_local = 0;
    ctx->cur_mem_usage_local = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->copy_13070_total_runtime = 0;
    ctx->copy_13070_runs = 0;
    ctx->map_12180_total_runtime = 0;
    ctx->map_12180_runs = 0;
    ctx->map_12229_total_runtime = 0;
    ctx->map_12229_runs = 0;
    ctx->map_12272_total_runtime = 0;
    ctx->map_12272_runs = 0;
    ctx->map_12337_total_runtime = 0;
    ctx->map_12337_runs = 0;
    ctx->map_12386_total_runtime = 0;
    ctx->map_12386_runs = 0;
    ctx->map_12422_total_runtime = 0;
    ctx->map_12422_runs = 0;
    ctx->map_12442_total_runtime = 0;
    ctx->map_12442_runs = 0;
    ctx->map_12478_total_runtime = 0;
    ctx->map_12478_runs = 0;
    ctx->map_12527_total_runtime = 0;
    ctx->map_12527_runs = 0;
    ctx->map_12562_total_runtime = 0;
    ctx->map_12562_runs = 0;
    ctx->map_transpose_f32_total_runtime = 0;
    ctx->map_transpose_f32_runs = 0;
    ctx->map_transpose_f32_low_height_total_runtime = 0;
    ctx->map_transpose_f32_low_height_runs = 0;
    ctx->map_transpose_f32_low_width_total_runtime = 0;
    ctx->map_transpose_f32_low_width_runs = 0;
    ctx->map_transpose_f32_small_total_runtime = 0;
    ctx->map_transpose_f32_small_runs = 0;
    ctx->replicate_13075_total_runtime = 0;
    ctx->replicate_13075_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    
    {
        ctx->copy_13070 = clCreateKernel(prog, "copy_13070", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "copy_13070");
    }
    {
        ctx->map_12180 = clCreateKernel(prog, "map_12180", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12180");
    }
    {
        ctx->map_12229 = clCreateKernel(prog, "map_12229", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12229");
    }
    {
        ctx->map_12272 = clCreateKernel(prog, "map_12272", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12272");
    }
    {
        ctx->map_12337 = clCreateKernel(prog, "map_12337", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12337");
    }
    {
        ctx->map_12386 = clCreateKernel(prog, "map_12386", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12386");
    }
    {
        ctx->map_12422 = clCreateKernel(prog, "map_12422", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12422");
    }
    {
        ctx->map_12442 = clCreateKernel(prog, "map_12442", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12442");
    }
    {
        ctx->map_12478 = clCreateKernel(prog, "map_12478", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12478");
    }
    {
        ctx->map_12527 = clCreateKernel(prog, "map_12527", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12527");
    }
    {
        ctx->map_12562 = clCreateKernel(prog, "map_12562", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_12562");
    }
    {
        ctx->map_transpose_f32 = clCreateKernel(prog, "map_transpose_f32",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_transpose_f32");
    }
    {
        ctx->map_transpose_f32_low_height = clCreateKernel(prog,
                                                           "map_transpose_f32_low_height",
                                                           &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "map_transpose_f32_low_height");
    }
    {
        ctx->map_transpose_f32_low_width = clCreateKernel(prog,
                                                          "map_transpose_f32_low_width",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "map_transpose_f32_low_width");
    }
    {
        ctx->map_transpose_f32_small = clCreateKernel(prog,
                                                      "map_transpose_f32_small",
                                                      &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_transpose_f32_small");
    }
    {
        ctx->replicate_13075 = clCreateKernel(prog, "replicate_13075", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "replicate_13075");
    }
    {
        cl_int success;
        
        ctx->static_array_13008.references = NULL;
        ctx->static_array_13008.size = 0;
        ctx->static_array_13008.mem = clCreateBuffer(ctx->opencl.ctx,
                                                     CL_MEM_READ_WRITE, (12 >
                                                                         0 ? 12 : 1) *
                                                     sizeof(float), NULL,
                                                     &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (12 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_13008.mem,
                                                          CL_TRUE, 0, 12 *
                                                          sizeof(float),
                                                          static_array_realtype_13089,
                                                          0, NULL, NULL));
    }
    {
        cl_int success;
        
        ctx->static_array_13025.references = NULL;
        ctx->static_array_13025.size = 0;
        ctx->static_array_13025.mem = clCreateBuffer(ctx->opencl.ctx,
                                                     CL_MEM_READ_WRITE, (16 >
                                                                         0 ? 16 : 1) *
                                                     sizeof(float), NULL,
                                                     &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (16 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_13025.mem,
                                                          CL_TRUE, 0, 16 *
                                                          sizeof(float),
                                                          static_array_realtype_13101,
                                                          0, NULL, NULL));
    }
    {
        cl_int success;
        
        ctx->static_array_13026.references = NULL;
        ctx->static_array_13026.size = 0;
        ctx->static_array_13026.mem = clCreateBuffer(ctx->opencl.ctx,
                                                     CL_MEM_READ_WRITE, (8 >
                                                                         0 ? 8 : 1) *
                                                     sizeof(float), NULL,
                                                     &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (8 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->static_array_13026.mem,
                                                          CL_TRUE, 0, 8 *
                                                          sizeof(float),
                                                          static_array_realtype_13102,
                                                          0, NULL, NULL));
    }
    ctx->sizes.mainzigroup_sizze_12174 = cfg->sizes[0];
    ctx->sizes.mainzigroup_sizze_12416 = cfg->sizes[1];
    ctx->sizes.mainzigroup_sizze_12436 = cfg->sizes[2];
    ctx->sizes.mainzigroup_sizze_12556 = cfg->sizes[3];
    ctx->sizes.mainzigroup_sizze_13073 = cfg->sizes[4];
    ctx->sizes.mainzigroup_sizze_13078 = cfg->sizes[5];
    ctx->sizes.mainzitile_sizze_12623 = cfg->sizes[6];
    ctx->sizes.mainzitile_sizze_12648 = cfg->sizes[7];
    ctx->sizes.mainzitile_sizze_12673 = cfg->sizes[8];
    ctx->sizes.mainzitile_sizze_12708 = cfg->sizes[9];
    ctx->sizes.mainzitile_sizze_12743 = cfg->sizes[10];
    ctx->sizes.mainzitile_sizze_12778 = cfg->sizes[11];
    return 0;
}
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program,
                                   required_types);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
struct futhark_context *futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                                               cl_command_queue queue)
{
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl_with_command_queue(&ctx->opencl, queue,
                                                      opencl_program,
                                                      required_types);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(clFinish(ctx->opencl.queue));
    return ctx->error != NULL;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(opencl_free_all(&ctx->opencl));
    return ctx->error != NULL;
}
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx)
{
    return ctx->opencl.queue;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED_OR_RETURN(opencl_free(&ctx->opencl, block->mem,
                                                 block->desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'device'",
              ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    OPENCL_SUCCEED_OR_RETURN(opencl_alloc(&ctx->opencl, size, desc,
                                          &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref_local(struct futhark_context *ctx,
                                struct memblock_local *block, const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'local'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_local -= block->size;
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_local);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_local(struct futhark_context *ctx,
                                struct memblock_local *block, int64_t size,
                                const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'local'",
              ctx->cur_mem_usage_local);
    
    int ret = memblock_unref_local(ctx, block, desc);
    
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_local += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'local'",
                (long long) ctx->cur_mem_usage_local);
    if (ctx->cur_mem_usage_local > ctx->peak_mem_usage_local) {
        ctx->peak_mem_usage_local = ctx->cur_mem_usage_local;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set_local(struct futhark_context *ctx,
                              struct memblock_local *lhs,
                              struct memblock_local *rhs, const char *lhs_desc)
{
    int ret = memblock_unref_local(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory) {
        fprintf(stderr, "Peak memory usage for space 'device': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for space 'local': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_local);
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->debugging) {
        fprintf(stderr,
                "Kernel copy_13070                   executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->copy_13070_runs, (long) ctx->copy_13070_total_runtime /
                (ctx->copy_13070_runs != 0 ? ctx->copy_13070_runs : 1),
                (long) ctx->copy_13070_total_runtime);
        ctx->total_runtime += ctx->copy_13070_total_runtime;
        ctx->total_runs += ctx->copy_13070_runs;
        fprintf(stderr,
                "Kernel map_12180                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12180_runs, (long) ctx->map_12180_total_runtime /
                (ctx->map_12180_runs != 0 ? ctx->map_12180_runs : 1),
                (long) ctx->map_12180_total_runtime);
        ctx->total_runtime += ctx->map_12180_total_runtime;
        ctx->total_runs += ctx->map_12180_runs;
        fprintf(stderr,
                "Kernel map_12229                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12229_runs, (long) ctx->map_12229_total_runtime /
                (ctx->map_12229_runs != 0 ? ctx->map_12229_runs : 1),
                (long) ctx->map_12229_total_runtime);
        ctx->total_runtime += ctx->map_12229_total_runtime;
        ctx->total_runs += ctx->map_12229_runs;
        fprintf(stderr,
                "Kernel map_12272                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12272_runs, (long) ctx->map_12272_total_runtime /
                (ctx->map_12272_runs != 0 ? ctx->map_12272_runs : 1),
                (long) ctx->map_12272_total_runtime);
        ctx->total_runtime += ctx->map_12272_total_runtime;
        ctx->total_runs += ctx->map_12272_runs;
        fprintf(stderr,
                "Kernel map_12337                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12337_runs, (long) ctx->map_12337_total_runtime /
                (ctx->map_12337_runs != 0 ? ctx->map_12337_runs : 1),
                (long) ctx->map_12337_total_runtime);
        ctx->total_runtime += ctx->map_12337_total_runtime;
        ctx->total_runs += ctx->map_12337_runs;
        fprintf(stderr,
                "Kernel map_12386                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12386_runs, (long) ctx->map_12386_total_runtime /
                (ctx->map_12386_runs != 0 ? ctx->map_12386_runs : 1),
                (long) ctx->map_12386_total_runtime);
        ctx->total_runtime += ctx->map_12386_total_runtime;
        ctx->total_runs += ctx->map_12386_runs;
        fprintf(stderr,
                "Kernel map_12422                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12422_runs, (long) ctx->map_12422_total_runtime /
                (ctx->map_12422_runs != 0 ? ctx->map_12422_runs : 1),
                (long) ctx->map_12422_total_runtime);
        ctx->total_runtime += ctx->map_12422_total_runtime;
        ctx->total_runs += ctx->map_12422_runs;
        fprintf(stderr,
                "Kernel map_12442                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12442_runs, (long) ctx->map_12442_total_runtime /
                (ctx->map_12442_runs != 0 ? ctx->map_12442_runs : 1),
                (long) ctx->map_12442_total_runtime);
        ctx->total_runtime += ctx->map_12442_total_runtime;
        ctx->total_runs += ctx->map_12442_runs;
        fprintf(stderr,
                "Kernel map_12478                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12478_runs, (long) ctx->map_12478_total_runtime /
                (ctx->map_12478_runs != 0 ? ctx->map_12478_runs : 1),
                (long) ctx->map_12478_total_runtime);
        ctx->total_runtime += ctx->map_12478_total_runtime;
        ctx->total_runs += ctx->map_12478_runs;
        fprintf(stderr,
                "Kernel map_12527                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12527_runs, (long) ctx->map_12527_total_runtime /
                (ctx->map_12527_runs != 0 ? ctx->map_12527_runs : 1),
                (long) ctx->map_12527_total_runtime);
        ctx->total_runtime += ctx->map_12527_total_runtime;
        ctx->total_runs += ctx->map_12527_runs;
        fprintf(stderr,
                "Kernel map_12562                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_12562_runs, (long) ctx->map_12562_total_runtime /
                (ctx->map_12562_runs != 0 ? ctx->map_12562_runs : 1),
                (long) ctx->map_12562_total_runtime);
        ctx->total_runtime += ctx->map_12562_total_runtime;
        ctx->total_runs += ctx->map_12562_runs;
        fprintf(stderr,
                "Kernel map_transpose_f32            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_transpose_f32_runs,
                (long) ctx->map_transpose_f32_total_runtime /
                (ctx->map_transpose_f32_runs !=
                 0 ? ctx->map_transpose_f32_runs : 1),
                (long) ctx->map_transpose_f32_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_runs;
        fprintf(stderr,
                "Kernel map_transpose_f32_low_height executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_transpose_f32_low_height_runs,
                (long) ctx->map_transpose_f32_low_height_total_runtime /
                (ctx->map_transpose_f32_low_height_runs !=
                 0 ? ctx->map_transpose_f32_low_height_runs : 1),
                (long) ctx->map_transpose_f32_low_height_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_low_height_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_low_height_runs;
        fprintf(stderr,
                "Kernel map_transpose_f32_low_width  executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_transpose_f32_low_width_runs,
                (long) ctx->map_transpose_f32_low_width_total_runtime /
                (ctx->map_transpose_f32_low_width_runs !=
                 0 ? ctx->map_transpose_f32_low_width_runs : 1),
                (long) ctx->map_transpose_f32_low_width_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_low_width_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_low_width_runs;
        fprintf(stderr,
                "Kernel map_transpose_f32_small      executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_transpose_f32_small_runs,
                (long) ctx->map_transpose_f32_small_total_runtime /
                (ctx->map_transpose_f32_small_runs !=
                 0 ? ctx->map_transpose_f32_small_runs : 1),
                (long) ctx->map_transpose_f32_small_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_small_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_small_runs;
        fprintf(stderr,
                "Kernel replicate_13075              executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->replicate_13075_runs,
                (long) ctx->replicate_13075_total_runtime /
                (ctx->replicate_13075_runs !=
                 0 ? ctx->replicate_13075_runs : 1),
                (long) ctx->replicate_13075_total_runtime);
        ctx->total_runtime += ctx->replicate_13075_total_runtime;
        ctx->total_runs += ctx->replicate_13075_runs;
        if (ctx->debugging)
            fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_main(struct futhark_context *ctx,
                       int64_t *out_out_memsizze_13080,
                       struct memblock_device *out_mem_p_13081,
                       int32_t *out_out_arrsizze_13082,
                       int32_t *out_out_arrsizze_13083,
                       int64_t image_mem_sizze_12813,
                       struct memblock_device image_mem_12814,
                       int64_t kernel_mem_sizze_12815,
                       struct memblock_device kernel_mem_12816,
                       int32_t sizze_11973, int32_t sizze_11974,
                       int32_t sizze_11975, int32_t sizze_11976);
static int futrts__map_transpose_f32(struct futhark_context *ctx,
                                     struct memblock_device destmem_0,
                                     int32_t destoffset_1,
                                     struct memblock_device srcmem_2,
                                     int32_t srcoffset_3, int32_t num_arrays_4,
                                     int32_t x_elems_5, int32_t y_elems_6,
                                     int32_t in_elems_7, int32_t out_elems_8);
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
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
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
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static int futrts_main(struct futhark_context *ctx,
                       int64_t *out_out_memsizze_13080,
                       struct memblock_device *out_mem_p_13081,
                       int32_t *out_out_arrsizze_13082,
                       int32_t *out_out_arrsizze_13083,
                       int64_t image_mem_sizze_12813,
                       struct memblock_device image_mem_12814,
                       int64_t kernel_mem_sizze_12815,
                       struct memblock_device kernel_mem_12816,
                       int32_t sizze_11973, int32_t sizze_11974,
                       int32_t sizze_11975, int32_t sizze_11976)
{
    int64_t out_memsizze_13003;
    struct memblock_device out_mem_13002;
    
    out_mem_13002.references = NULL;
    
    int32_t out_arrsizze_13004;
    int32_t out_arrsizze_13005;
    int32_t range_end_11979 = 1 + sizze_11973;
    bool bounds_invalid_upwards_11980 = slt32(range_end_11979, 0);
    int32_t distance_11981 = 1 + range_end_11979;
    int32_t num_elems_11982;
    
    if (bounds_invalid_upwards_11980) {
        num_elems_11982 = 0;
    } else {
        num_elems_11982 = distance_11981;
    }
    
    int32_t range_end_11984 = 1 + sizze_11974;
    bool bounds_invalid_upwards_11985 = slt32(range_end_11984, 0);
    int32_t distance_11986 = 1 + range_end_11984;
    int32_t num_elems_11987;
    
    if (bounds_invalid_upwards_11985) {
        num_elems_11987 = 0;
    } else {
        num_elems_11987 = distance_11986;
    }
    
    int32_t nesting_sizze_12173 = num_elems_11982 * num_elems_11987;
    int32_t group_sizze_12175;
    
    group_sizze_12175 = ctx->sizes.mainzigroup_sizze_12174;
    
    int32_t y_12176 = group_sizze_12175 - 1;
    int32_t x_12177 = nesting_sizze_12173 + y_12176;
    int32_t num_groups_12178 = squot32(x_12177, group_sizze_12175);
    int32_t num_threads_12179 = group_sizze_12175 * num_groups_12178;
    int64_t binop_x_12819 = sext_i32_i64(nesting_sizze_12173);
    int64_t bytes_12817 = 4 * binop_x_12819;
    struct memblock_device mem_12820;
    
    mem_12820.references = NULL;
    if (memblock_alloc_device(ctx, &mem_12820, bytes_12817, "mem_12820"))
        return 1;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12180, 0,
                                            sizeof(sizze_11974), &sizze_11974));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12180, 1,
                                            sizeof(range_end_11979),
                                            &range_end_11979));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12180, 2,
                                            sizeof(num_elems_11982),
                                            &num_elems_11982));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12180, 3,
                                            sizeof(range_end_11984),
                                            &range_end_11984));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12180, 4,
                                            sizeof(num_elems_11987),
                                            &num_elems_11987));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12180, 5,
                                            sizeof(image_mem_12814.mem),
                                            &image_mem_12814.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12180, 6,
                                            sizeof(mem_12820.mem),
                                            &mem_12820.mem));
    if (1 * (num_groups_12178 * group_sizze_12175) != 0) {
        const size_t global_work_sizze_13084[1] = {num_groups_12178 *
                     group_sizze_12175};
        const size_t local_work_sizze_13088[1] = {group_sizze_12175};
        int64_t time_start_13085 = 0, time_end_13086 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_12180");
            fprintf(stderr, "%zu", global_work_sizze_13084[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_13088[0]);
            fprintf(stderr, "].\n");
            time_start_13085 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_12180, 1, NULL,
                                                        global_work_sizze_13084,
                                                        local_work_sizze_13088,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_13086 = get_wall_time();
            
            long time_diff_13087 = time_end_13086 - time_start_13085;
            
            ctx->map_12180_total_runtime += time_diff_13087;
            ctx->map_12180_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12180",
                    time_diff_13087);
        }
    }
    
    int32_t res_12018 = sdiv32(sizze_11974, 2);
    int32_t res_12019 = sdiv32(sizze_11973, 2);
    bool dim_match_12020 = 3 == sizze_11975;
    bool dim_match_12021 = 3 == sizze_11976;
    bool match_12022 = dim_match_12020 && dim_match_12021;
    bool empty_or_match_cert_12023;
    
    if (!match_12022) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "winograd.fut:61:3-74:10 -> winograd.fut:67:20-41",
                               "function arguments of wrong shape");
        if (memblock_unref_device(ctx, &mem_12820, "mem_12820") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_13002, "out_mem_13002") != 0)
            return 1;
        return 1;
    }
    
    struct memblock_device mem_12823;
    
    mem_12823.references = NULL;
    if (memblock_alloc_device(ctx, &mem_12823, 48, "mem_12823"))
        return 1;
    
    struct memblock_device static_array_13008 = ctx->static_array_13008;
    
    if (12 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_13008.mem,
                                                     mem_12823.mem, 0, 0, 12 *
                                                     sizeof(float), 0, NULL,
                                                     NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    int32_t tile_sizze_12624;
    
    tile_sizze_12624 = ctx->sizes.mainzitile_sizze_12623;
    
    int32_t tiled_group_sizze_12625 = tile_sizze_12624 * tile_sizze_12624;
    int32_t y_12628 = tile_sizze_12624 - 1;
    int32_t x_12629 = 4 + y_12628;
    int32_t groups_in_dim_12630 = squot32(x_12629, tile_sizze_12624);
    int32_t x_12632 = 3 + y_12628;
    int32_t groups_in_dim_12633 = squot32(x_12632, tile_sizze_12624);
    int32_t num_groups_12635 = groups_in_dim_12630 * groups_in_dim_12633;
    int32_t num_threads_12636 = tiled_group_sizze_12625 * num_groups_12635;
    struct memblock_device mem_12834;
    
    mem_12834.references = NULL;
    if (memblock_alloc_device(ctx, &mem_12834, 48, "mem_12834"))
        return 1;
    
    int64_t binop_x_12826 = sext_i32_i64(tiled_group_sizze_12625);
    int64_t bytes_12824 = 4 * binop_x_12826;
    struct memblock_local mem_12827;
    
    mem_12827.references = NULL;
    
    struct memblock_local mem_12831;
    
    mem_12831.references = NULL;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12272, 0, bytes_12824,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12272, 1, bytes_12824,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12272, 2,
                                            sizeof(sizze_11976), &sizze_11976));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12272, 3,
                                            sizeof(kernel_mem_12816.mem),
                                            &kernel_mem_12816.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12272, 4,
                                            sizeof(mem_12823.mem),
                                            &mem_12823.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12272, 5,
                                            sizeof(mem_12834.mem),
                                            &mem_12834.mem));
    if (1 * (num_groups_12635 * tiled_group_sizze_12625) != 0) {
        const size_t global_work_sizze_13090[1] = {num_groups_12635 *
                     tiled_group_sizze_12625};
        const size_t local_work_sizze_13094[1] = {tiled_group_sizze_12625};
        int64_t time_start_13091 = 0, time_end_13092 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_12272");
            fprintf(stderr, "%zu", global_work_sizze_13090[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_13094[0]);
            fprintf(stderr, "].\n");
            time_start_13091 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_12272, 1, NULL,
                                                        global_work_sizze_13090,
                                                        local_work_sizze_13094,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_13092 = get_wall_time();
            
            long time_diff_13093 = time_end_13092 - time_start_13091;
            
            ctx->map_12272_total_runtime += time_diff_13093;
            ctx->map_12272_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12272",
                    time_diff_13093);
        }
    }
    if (memblock_unref_local(ctx, &mem_12827, "mem_12827") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_12831, "mem_12831") != 0)
        return 1;
    
    struct memblock_device mem_12837;
    
    mem_12837.references = NULL;
    if (memblock_alloc_device(ctx, &mem_12837, 48, "mem_12837"))
        return 1;
    
    int call_ret_13095 = futrts__map_transpose_f32(ctx, mem_12837, 0, mem_12823,
                                                   0, 1, 3, 4, 12, 12);
    
    assert(call_ret_13095 == 0);
    if (memblock_unref_device(ctx, &mem_12823, "mem_12823") != 0)
        return 1;
    
    int32_t tile_sizze_12649;
    
    tile_sizze_12649 = ctx->sizes.mainzitile_sizze_12648;
    
    int32_t tiled_group_sizze_12650 = tile_sizze_12649 * tile_sizze_12649;
    int32_t y_12653 = tile_sizze_12649 - 1;
    int32_t x_12654 = 4 + y_12653;
    int32_t groups_in_dim_12655 = squot32(x_12654, tile_sizze_12649);
    int32_t num_groups_12660 = groups_in_dim_12655 * groups_in_dim_12655;
    int32_t num_threads_12661 = tiled_group_sizze_12650 * num_groups_12660;
    struct memblock_device mem_12848;
    
    mem_12848.references = NULL;
    if (memblock_alloc_device(ctx, &mem_12848, 64, "mem_12848"))
        return 1;
    
    int64_t binop_x_12840 = sext_i32_i64(tiled_group_sizze_12650);
    int64_t bytes_12838 = 4 * binop_x_12840;
    struct memblock_local mem_12841;
    
    mem_12841.references = NULL;
    
    struct memblock_local mem_12845;
    
    mem_12845.references = NULL;
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12229, 0, bytes_12838,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12229, 1, bytes_12838,
                                            NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12229, 2,
                                            sizeof(mem_12834.mem),
                                            &mem_12834.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12229, 3,
                                            sizeof(mem_12837.mem),
                                            &mem_12837.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12229, 4,
                                            sizeof(mem_12848.mem),
                                            &mem_12848.mem));
    if (1 * (num_groups_12660 * tiled_group_sizze_12650) != 0) {
        const size_t global_work_sizze_13096[1] = {num_groups_12660 *
                     tiled_group_sizze_12650};
        const size_t local_work_sizze_13100[1] = {tiled_group_sizze_12650};
        int64_t time_start_13097 = 0, time_end_13098 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_12229");
            fprintf(stderr, "%zu", global_work_sizze_13096[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_13100[0]);
            fprintf(stderr, "].\n");
            time_start_13097 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->map_12229, 1, NULL,
                                                        global_work_sizze_13096,
                                                        local_work_sizze_13100,
                                                        0, NULL, NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_13098 = get_wall_time();
            
            long time_diff_13099 = time_end_13098 - time_start_13097;
            
            ctx->map_12229_total_runtime += time_diff_13099;
            ctx->map_12229_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12229",
                    time_diff_13099);
        }
    }
    if (memblock_unref_device(ctx, &mem_12834, "mem_12834") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12837, "mem_12837") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_12841, "mem_12841") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_12845, "mem_12845") != 0)
        return 1;
    
    int32_t x_12048 = sizze_11973 * sizze_11974;
    int32_t x_12049 = smod32(x_12048, 4);
    bool cond_12050 = x_12049 == 0;
    int32_t arg_12051 = 2 * res_12019;
    int32_t res_12052 = sdiv32(arg_12051, 2);
    bool bounds_invalid_upwards_12053 = slt32(res_12052, 0);
    int32_t arg_12054 = 2 * res_12018;
    int32_t res_12055 = sdiv32(arg_12054, 2);
    int32_t flat_dim_12056 = 2 * res_12055;
    bool bounds_invalid_upwards_12057 = slt32(res_12055, 0);
    struct memblock_device mem_12851;
    
    mem_12851.references = NULL;
    if (memblock_alloc_device(ctx, &mem_12851, 64, "mem_12851"))
        return 1;
    
    struct memblock_device static_array_13025 = ctx->static_array_13025;
    
    if (16 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_13025.mem,
                                                     mem_12851.mem, 0, 0, 16 *
                                                     sizeof(float), 0, NULL,
                                                     NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_12854;
    
    mem_12854.references = NULL;
    if (memblock_alloc_device(ctx, &mem_12854, 32, "mem_12854"))
        return 1;
    
    struct memblock_device static_array_13026 = ctx->static_array_13026;
    
    if (8 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     static_array_13026.mem,
                                                     mem_12854.mem, 0, 0, 8 *
                                                     sizeof(float), 0, NULL,
                                                     NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    int32_t arg_12060 = num_elems_11982 - 2;
    int32_t arg_12061 = num_elems_11987 - 2;
    int32_t flat_dim_12062 = res_12052 * flat_dim_12056;
    int32_t flat_dim_12063 = 2 * flat_dim_12062;
    int32_t sizze_12064;
    
    if (cond_12050) {
        sizze_12064 = arg_12060;
    } else {
        sizze_12064 = 1;
    }
    
    int32_t sizze_12065;
    
    if (cond_12050) {
        sizze_12065 = arg_12061;
    } else {
        sizze_12065 = 1;
    }
    
    int32_t nesting_sizze_12555 = res_12052 * res_12055;
    int32_t group_sizze_12557;
    
    group_sizze_12557 = ctx->sizes.mainzigroup_sizze_12556;
    
    int32_t y_12558 = group_sizze_12557 - 1;
    int32_t x_12559 = nesting_sizze_12555 + y_12558;
    int32_t binop_x_12869 = 16 * res_12052;
    int32_t convop_x_12870 = res_12055 * binop_x_12869;
    int64_t binop_x_12871 = sext_i32_i64(convop_x_12870);
    int64_t bytes_12868 = 4 * binop_x_12871;
    int32_t binop_x_12878 = 4 * nesting_sizze_12555;
    int32_t convop_x_12879 = 4 * binop_x_12878;
    int64_t binop_x_12880 = sext_i32_i64(convop_x_12879);
    int64_t bytes_12876 = 4 * binop_x_12880;
    int32_t binop_x_12883 = 4 * res_12052;
    int32_t binop_x_12884 = res_12055 * binop_x_12883;
    int32_t convop_x_12885 = 4 * binop_x_12884;
    int64_t binop_x_12886 = sext_i32_i64(convop_x_12885);
    int64_t bytes_12882 = 4 * binop_x_12886;
    int32_t binop_x_12957 = 2 * nesting_sizze_12555;
    int32_t convop_x_12958 = 4 * binop_x_12957;
    int64_t binop_x_12959 = sext_i32_i64(convop_x_12958);
    int64_t bytes_12955 = 4 * binop_x_12959;
    int32_t convop_x_12972 = 2 * binop_x_12957;
    int64_t binop_x_12973 = sext_i32_i64(convop_x_12972);
    int64_t bytes_12969 = 4 * binop_x_12973;
    int32_t convop_x_12982 = sizze_12064 * sizze_12065;
    int64_t binop_x_12983 = sext_i32_i64(convop_x_12982);
    int64_t bytes_12981 = 4 * binop_x_12983;
    int32_t tile_sizze_12674;
    
    tile_sizze_12674 = ctx->sizes.mainzitile_sizze_12673;
    
    int32_t tiled_group_sizze_12675 = tile_sizze_12674 * tile_sizze_12674;
    int64_t binop_x_12890 = sext_i32_i64(tiled_group_sizze_12675);
    int64_t bytes_12888 = 4 * binop_x_12890;
    int32_t tile_sizze_12709;
    
    tile_sizze_12709 = ctx->sizes.mainzitile_sizze_12708;
    
    int32_t tiled_group_sizze_12710 = tile_sizze_12709 * tile_sizze_12709;
    int64_t binop_x_12904 = sext_i32_i64(tiled_group_sizze_12710);
    int64_t bytes_12902 = 4 * binop_x_12904;
    int32_t tile_sizze_12744;
    
    tile_sizze_12744 = ctx->sizes.mainzitile_sizze_12743;
    
    int32_t tiled_group_sizze_12745 = tile_sizze_12744 * tile_sizze_12744;
    int64_t binop_x_12949 = sext_i32_i64(tiled_group_sizze_12745);
    int64_t bytes_12947 = 4 * binop_x_12949;
    int32_t tile_sizze_12779;
    
    tile_sizze_12779 = ctx->sizes.mainzitile_sizze_12778;
    
    int32_t tiled_group_sizze_12780 = tile_sizze_12779 * tile_sizze_12779;
    int64_t binop_x_12963 = sext_i32_i64(tiled_group_sizze_12780);
    int64_t bytes_12961 = 4 * binop_x_12963;
    struct memblock_device res_mem_12991;
    
    res_mem_12991.references = NULL;
    if (cond_12050) {
        bool eq_x_zz_12067 = 0 == res_12052;
        bool not_p_12068 = !bounds_invalid_upwards_12053;
        bool p_and_eq_x_y_12069 = eq_x_zz_12067 && not_p_12068;
        bool dim_zzero_12070 = bounds_invalid_upwards_12053 ||
             p_and_eq_x_y_12069;
        bool both_empty_12071 = eq_x_zz_12067 && dim_zzero_12070;
        bool eq_x_y_12072 = res_12052 == 0;
        bool p_and_eq_x_y_12073 = bounds_invalid_upwards_12053 && eq_x_y_12072;
        bool dim_match_12074 = not_p_12068 || p_and_eq_x_y_12073;
        bool empty_or_match_12075 = both_empty_12071 || dim_match_12074;
        bool empty_or_match_cert_12076;
        
        if (!empty_or_match_12075) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "winograd.fut:61:3-74:10 -> winograd.fut:71:9-74 -> winograd.fut:43:12-32 -> /futlib/array.fut:113:6-23 -> /futlib/array.fut:109:30-43 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", res_12052, "]", "intrinsics.i32");
            if (memblock_unref_device(ctx, &res_mem_12991, "res_mem_12991") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12854, "mem_12854") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12851, "mem_12851") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12845, "mem_12845") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12841, "mem_12841") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12848, "mem_12848") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12837, "mem_12837") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12831, "mem_12831") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12827, "mem_12827") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12834, "mem_12834") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12823, "mem_12823") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12820, "mem_12820") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_13002, "out_mem_13002") !=
                0)
                return 1;
            return 1;
        }
        
        bool eq_x_zz_12078 = 0 == res_12055;
        bool not_p_12079 = !bounds_invalid_upwards_12057;
        bool p_and_eq_x_y_12080 = eq_x_zz_12078 && not_p_12079;
        bool dim_zzero_12081 = bounds_invalid_upwards_12057 ||
             p_and_eq_x_y_12080;
        bool both_empty_12082 = eq_x_zz_12078 && dim_zzero_12081;
        bool eq_x_y_12083 = res_12055 == 0;
        bool p_and_eq_x_y_12084 = bounds_invalid_upwards_12057 && eq_x_y_12083;
        bool dim_match_12085 = not_p_12079 || p_and_eq_x_y_12084;
        bool empty_or_match_12086 = both_empty_12082 || dim_match_12085;
        bool empty_or_match_cert_12087;
        
        if (!empty_or_match_12086) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "winograd.fut:61:3-74:10 -> winograd.fut:71:9-74 -> winograd.fut:38:7-43:33 -> winograd.fut:42:18-38 -> /futlib/array.fut:113:6-23 -> /futlib/array.fut:109:30-43 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", res_12055, "]", "intrinsics.i32");
            if (memblock_unref_device(ctx, &res_mem_12991, "res_mem_12991") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12854, "mem_12854") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12851, "mem_12851") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12845, "mem_12845") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12841, "mem_12841") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12848, "mem_12848") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12837, "mem_12837") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12831, "mem_12831") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12827, "mem_12827") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12834, "mem_12834") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12823, "mem_12823") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12820, "mem_12820") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_13002, "out_mem_13002") !=
                0)
                return 1;
            return 1;
        }
        
        int32_t num_groups_12560 = squot32(x_12559, group_sizze_12557);
        int32_t num_threads_12561 = group_sizze_12557 * num_groups_12560;
        int32_t y_12598 = smod32(num_elems_11982, num_threads_12561);
        int32_t x_12599 = num_threads_12561 - y_12598;
        int32_t y_12600 = smod32(x_12599, num_threads_12561);
        int32_t padded_sizze_12601 = num_elems_11982 + y_12600;
        int32_t per_chunk_12603 = squot32(padded_sizze_12601,
                                          num_threads_12561);
        int32_t convop_x_12856 = num_elems_11987 * y_12600;
        int64_t binop_x_12857 = sext_i32_i64(convop_x_12856);
        int64_t bytes_12855 = 4 * binop_x_12857;
        struct memblock_device mem_12858;
        
        mem_12858.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12858, bytes_12855, "mem_12858"))
            return 1;
        
        int32_t convop_x_12860 = num_elems_11987 * padded_sizze_12601;
        int64_t binop_x_12861 = sext_i32_i64(convop_x_12860);
        int64_t bytes_12859 = 4 * binop_x_12861;
        struct memblock_device mem_12862;
        
        mem_12862.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12862, bytes_12859, "mem_12862"))
            return 1;
        
        int32_t tmp_offs_13027 = 0;
        
        if (num_elems_11982 * num_elems_11987 * sizeof(float) > 0) {
            OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                         mem_12820.mem,
                                                         mem_12862.mem, 0,
                                                         num_elems_11987 *
                                                         tmp_offs_13027 * 4,
                                                         num_elems_11982 *
                                                         num_elems_11987 *
                                                         sizeof(float), 0, NULL,
                                                         NULL));
            if (ctx->debugging)
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
        }
        tmp_offs_13027 += num_elems_11982;
        if (y_12600 * num_elems_11987 * sizeof(float) > 0) {
            OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                         mem_12858.mem,
                                                         mem_12862.mem, 0,
                                                         num_elems_11987 *
                                                         tmp_offs_13027 * 4,
                                                         y_12600 *
                                                         num_elems_11987 *
                                                         sizeof(float), 0, NULL,
                                                         NULL));
            if (ctx->debugging)
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
        }
        tmp_offs_13027 += y_12600;
        if (memblock_unref_device(ctx, &mem_12858, "mem_12858") != 0)
            return 1;
        
        int32_t binop_x_12864 = num_elems_11987 * per_chunk_12603;
        int32_t convop_x_12865 = num_threads_12561 * binop_x_12864;
        int64_t binop_x_12866 = sext_i32_i64(convop_x_12865);
        int64_t bytes_12863 = 4 * binop_x_12866;
        struct memblock_device mem_12867;
        
        mem_12867.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12867, bytes_12863, "mem_12867"))
            return 1;
        
        int call_ret_13103 = futrts__map_transpose_f32(ctx, mem_12867, 0,
                                                       mem_12862, 0, 1,
                                                       per_chunk_12603 *
                                                       num_elems_11987,
                                                       num_threads_12561,
                                                       num_threads_12561 *
                                                       per_chunk_12603 *
                                                       num_elems_11987,
                                                       num_threads_12561 *
                                                       per_chunk_12603 *
                                                       num_elems_11987);
        
        assert(call_ret_13103 == 0);
        if (memblock_unref_device(ctx, &mem_12862, "mem_12862") != 0)
            return 1;
        
        struct memblock_device mem_12872;
        
        mem_12872.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12872, bytes_12868, "mem_12872"))
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12562, 0,
                                                sizeof(num_elems_11987),
                                                &num_elems_11987));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12562, 1,
                                                sizeof(res_12052), &res_12052));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12562, 2,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12562, 3,
                                                sizeof(num_threads_12561),
                                                &num_threads_12561));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12562, 4,
                                                sizeof(per_chunk_12603),
                                                &per_chunk_12603));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12562, 5,
                                                sizeof(mem_12867.mem),
                                                &mem_12867.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12562, 6,
                                                sizeof(mem_12872.mem),
                                                &mem_12872.mem));
        if (1 * (num_groups_12560 * group_sizze_12557) != 0) {
            const size_t global_work_sizze_13104[1] = {num_groups_12560 *
                         group_sizze_12557};
            const size_t local_work_sizze_13108[1] = {group_sizze_12557};
            int64_t time_start_13105 = 0, time_end_13106 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_12562");
                fprintf(stderr, "%zu", global_work_sizze_13104[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13108[0]);
                fprintf(stderr, "].\n");
                time_start_13105 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->map_12562, 1,
                                                            NULL,
                                                            global_work_sizze_13104,
                                                            local_work_sizze_13108,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13106 = get_wall_time();
                
                long time_diff_13107 = time_end_13106 - time_start_13105;
                
                ctx->map_12562_total_runtime += time_diff_13107;
                ctx->map_12562_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12562",
                        time_diff_13107);
            }
        }
        if (memblock_unref_device(ctx, &mem_12867, "mem_12867") != 0)
            return 1;
        
        struct memblock_device mem_12875;
        
        mem_12875.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12875, 64, "mem_12875"))
            return 1;
        
        int call_ret_13109 = futrts__map_transpose_f32(ctx, mem_12875, 0,
                                                       mem_12851, 0, 1, 4, 4,
                                                       16, 16);
        
        assert(call_ret_13109 == 0);
        
        struct memblock_device mem_12881;
        
        mem_12881.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12881, bytes_12876, "mem_12881"))
            return 1;
        
        int call_ret_13110 = futrts__map_transpose_f32(ctx, mem_12881, 0,
                                                       mem_12872, 0, 1,
                                                       res_12052 * res_12055,
                                                       16, res_12052 *
                                                       res_12055 * 4 * 4,
                                                       res_12052 * res_12055 *
                                                       4 * 4);
        
        assert(call_ret_13110 == 0);
        if (memblock_unref_device(ctx, &mem_12872, "mem_12872") != 0)
            return 1;
        
        struct memblock_device mem_12887;
        
        mem_12887.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12887, bytes_12882, "mem_12887"))
            return 1;
        
        int call_ret_13111 = futrts__map_transpose_f32(ctx, mem_12887, 0,
                                                       mem_12881, 0, 1, 4,
                                                       res_12052 * res_12055 *
                                                       4, res_12052 *
                                                       res_12055 * 4 * 4,
                                                       res_12052 * res_12055 *
                                                       4 * 4);
        
        assert(call_ret_13111 == 0);
        if (memblock_unref_device(ctx, &mem_12881, "mem_12881") != 0)
            return 1;
        
        int32_t y_12686 = tile_sizze_12674 - 1;
        int32_t x_12687 = 4 + y_12686;
        int32_t groups_in_dim_12688 = squot32(x_12687, tile_sizze_12674);
        int32_t y_12693 = groups_in_dim_12688 * groups_in_dim_12688;
        int32_t y_12694 = res_12055 * y_12693;
        int32_t num_groups_12695 = res_12052 * y_12694;
        int32_t num_threads_12696 = tiled_group_sizze_12675 * num_groups_12695;
        struct memblock_device mem_12901;
        
        mem_12901.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12901, bytes_12876, "mem_12901"))
            return 1;
        
        struct memblock_local mem_12891;
        
        mem_12891.references = NULL;
        
        struct memblock_local mem_12895;
        
        mem_12895.references = NULL;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12527, 0, bytes_12888,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12527, 1, bytes_12888,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12527, 2,
                                                sizeof(res_12052), &res_12052));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12527, 3,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12527, 4,
                                                sizeof(mem_12875.mem),
                                                &mem_12875.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12527, 5,
                                                sizeof(mem_12887.mem),
                                                &mem_12887.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12527, 6,
                                                sizeof(mem_12901.mem),
                                                &mem_12901.mem));
        if (1 * (num_groups_12695 * tiled_group_sizze_12675) != 0) {
            const size_t global_work_sizze_13112[1] = {num_groups_12695 *
                         tiled_group_sizze_12675};
            const size_t local_work_sizze_13116[1] = {tiled_group_sizze_12675};
            int64_t time_start_13113 = 0, time_end_13114 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_12527");
                fprintf(stderr, "%zu", global_work_sizze_13112[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13116[0]);
                fprintf(stderr, "].\n");
                time_start_13113 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->map_12527, 1,
                                                            NULL,
                                                            global_work_sizze_13112,
                                                            local_work_sizze_13116,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13114 = get_wall_time();
                
                long time_diff_13115 = time_end_13114 - time_start_13113;
                
                ctx->map_12527_total_runtime += time_diff_13115;
                ctx->map_12527_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12527",
                        time_diff_13115);
            }
        }
        if (memblock_unref_device(ctx, &mem_12887, "mem_12887") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12891, "mem_12891") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12895, "mem_12895") != 0)
            return 1;
        
        int32_t y_12721 = tile_sizze_12709 - 1;
        int32_t x_12722 = 4 + y_12721;
        int32_t groups_in_dim_12723 = squot32(x_12722, tile_sizze_12709);
        int32_t y_12728 = groups_in_dim_12723 * groups_in_dim_12723;
        int32_t y_12729 = res_12055 * y_12728;
        int32_t num_groups_12730 = res_12052 * y_12729;
        int32_t num_threads_12731 = tiled_group_sizze_12710 * num_groups_12730;
        struct memblock_device mem_12915;
        
        mem_12915.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12915, bytes_12876, "mem_12915"))
            return 1;
        
        struct memblock_local mem_12905;
        
        mem_12905.references = NULL;
        
        struct memblock_local mem_12909;
        
        mem_12909.references = NULL;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12478, 0, bytes_12902,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12478, 1, bytes_12902,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12478, 2,
                                                sizeof(res_12052), &res_12052));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12478, 3,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12478, 4,
                                                sizeof(mem_12875.mem),
                                                &mem_12875.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12478, 5,
                                                sizeof(mem_12901.mem),
                                                &mem_12901.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12478, 6,
                                                sizeof(mem_12915.mem),
                                                &mem_12915.mem));
        if (1 * (num_groups_12730 * tiled_group_sizze_12710) != 0) {
            const size_t global_work_sizze_13117[1] = {num_groups_12730 *
                         tiled_group_sizze_12710};
            const size_t local_work_sizze_13121[1] = {tiled_group_sizze_12710};
            int64_t time_start_13118 = 0, time_end_13119 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_12478");
                fprintf(stderr, "%zu", global_work_sizze_13117[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13121[0]);
                fprintf(stderr, "].\n");
                time_start_13118 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->map_12478, 1,
                                                            NULL,
                                                            global_work_sizze_13117,
                                                            local_work_sizze_13121,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13119 = get_wall_time();
                
                long time_diff_13120 = time_end_13119 - time_start_13118;
                
                ctx->map_12478_total_runtime += time_diff_13120;
                ctx->map_12478_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12478",
                        time_diff_13120);
            }
        }
        if (memblock_unref_device(ctx, &mem_12875, "mem_12875") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12901, "mem_12901") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12905, "mem_12905") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12909, "mem_12909") != 0)
            return 1;
        
        int32_t nesting_sizze_12434 = 16 * res_12055;
        int32_t nesting_sizze_12435 = res_12052 * nesting_sizze_12434;
        int32_t group_sizze_12437;
        
        group_sizze_12437 = ctx->sizes.mainzigroup_sizze_12436;
        
        int32_t y_12438 = group_sizze_12437 - 1;
        int32_t x_12439 = nesting_sizze_12435 + y_12438;
        int32_t num_groups_12440 = squot32(x_12439, group_sizze_12437);
        int32_t num_threads_12441 = group_sizze_12437 * num_groups_12440;
        struct memblock_device mem_12921;
        
        mem_12921.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12921, bytes_12876, "mem_12921"))
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12442, 0,
                                                sizeof(res_12052), &res_12052));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12442, 1,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12442, 2,
                                                sizeof(mem_12848.mem),
                                                &mem_12848.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12442, 3,
                                                sizeof(mem_12915.mem),
                                                &mem_12915.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12442, 4,
                                                sizeof(mem_12921.mem),
                                                &mem_12921.mem));
        if (1 * (num_groups_12440 * group_sizze_12437) != 0) {
            const size_t global_work_sizze_13122[1] = {num_groups_12440 *
                         group_sizze_12437};
            const size_t local_work_sizze_13126[1] = {group_sizze_12437};
            int64_t time_start_13123 = 0, time_end_13124 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_12442");
                fprintf(stderr, "%zu", global_work_sizze_13122[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13126[0]);
                fprintf(stderr, "].\n");
                time_start_13123 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->map_12442, 1,
                                                            NULL,
                                                            global_work_sizze_13122,
                                                            local_work_sizze_13126,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13124 = get_wall_time();
                
                long time_diff_13125 = time_end_13124 - time_start_13123;
                
                ctx->map_12442_total_runtime += time_diff_13125;
                ctx->map_12442_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12442",
                        time_diff_13125);
            }
        }
        if (memblock_unref_device(ctx, &mem_12915, "mem_12915") != 0)
            return 1;
        
        int32_t group_sizze_12417;
        
        group_sizze_12417 = ctx->sizes.mainzigroup_sizze_12416;
        
        int32_t y_12418 = group_sizze_12417 - 1;
        int32_t x_12419 = y_12418 + nesting_sizze_12555;
        int32_t num_groups_12420 = squot32(x_12419, group_sizze_12417);
        int32_t num_threads_12421 = group_sizze_12417 * num_groups_12420;
        struct memblock_device mem_12926;
        
        mem_12926.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12926, bytes_12868, "mem_12926"))
            return 1;
        
        int call_ret_13127 = futrts__map_transpose_f32(ctx, mem_12926, 0,
                                                       mem_12921, 0, 1, 16,
                                                       res_12052 * res_12055,
                                                       res_12052 * res_12055 *
                                                       4 * 4, res_12052 *
                                                       res_12055 * 4 * 4);
        
        assert(call_ret_13127 == 0);
        if (memblock_unref_device(ctx, &mem_12921, "mem_12921") != 0)
            return 1;
        
        struct memblock_device mem_12931;
        
        mem_12931.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12931, bytes_12868, "mem_12931"))
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12422, 0,
                                                sizeof(res_12052), &res_12052));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12422, 1,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12422, 2,
                                                sizeof(mem_12926.mem),
                                                &mem_12926.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12422, 3,
                                                sizeof(mem_12931.mem),
                                                &mem_12931.mem));
        if (1 * (num_groups_12420 * group_sizze_12417) != 0) {
            const size_t global_work_sizze_13128[1] = {num_groups_12420 *
                         group_sizze_12417};
            const size_t local_work_sizze_13132[1] = {group_sizze_12417};
            int64_t time_start_13129 = 0, time_end_13130 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_12422");
                fprintf(stderr, "%zu", global_work_sizze_13128[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13132[0]);
                fprintf(stderr, "].\n");
                time_start_13129 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->map_12422, 1,
                                                            NULL,
                                                            global_work_sizze_13128,
                                                            local_work_sizze_13132,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13130 = get_wall_time();
                
                long time_diff_13131 = time_end_13130 - time_start_13129;
                
                ctx->map_12422_total_runtime += time_diff_13131;
                ctx->map_12422_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12422",
                        time_diff_13131);
            }
        }
        if (memblock_unref_device(ctx, &mem_12926, "mem_12926") != 0)
            return 1;
        
        struct memblock_device mem_12934;
        
        mem_12934.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12934, 32, "mem_12934"))
            return 1;
        
        int call_ret_13133 = futrts__map_transpose_f32(ctx, mem_12934, 0,
                                                       mem_12854, 0, 1, 4, 2, 8,
                                                       8);
        
        assert(call_ret_13133 == 0);
        
        struct memblock_device mem_12940;
        
        mem_12940.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12940, bytes_12876, "mem_12940"))
            return 1;
        
        int call_ret_13134 = futrts__map_transpose_f32(ctx, mem_12940, 0,
                                                       mem_12931, 0, 1,
                                                       res_12052 * res_12055,
                                                       16, res_12052 *
                                                       res_12055 * 4 * 4,
                                                       res_12052 * res_12055 *
                                                       4 * 4);
        
        assert(call_ret_13134 == 0);
        if (memblock_unref_device(ctx, &mem_12931, "mem_12931") != 0)
            return 1;
        
        struct memblock_device mem_12946;
        
        mem_12946.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12946, bytes_12882, "mem_12946"))
            return 1;
        
        int call_ret_13135 = futrts__map_transpose_f32(ctx, mem_12946, 0,
                                                       mem_12940, 0, 1, 4,
                                                       res_12052 * res_12055 *
                                                       4, res_12052 *
                                                       res_12055 * 4 * 4,
                                                       res_12052 * res_12055 *
                                                       4 * 4);
        
        assert(call_ret_13135 == 0);
        if (memblock_unref_device(ctx, &mem_12940, "mem_12940") != 0)
            return 1;
        
        int32_t y_12756 = tile_sizze_12744 - 1;
        int32_t x_12757 = 2 + y_12756;
        int32_t groups_in_dim_12758 = squot32(x_12757, tile_sizze_12744);
        int32_t x_12760 = 4 + y_12756;
        int32_t groups_in_dim_12761 = squot32(x_12760, tile_sizze_12744);
        int32_t y_12763 = groups_in_dim_12758 * groups_in_dim_12761;
        int32_t y_12764 = res_12055 * y_12763;
        int32_t num_groups_12765 = res_12052 * y_12764;
        int32_t num_threads_12766 = tiled_group_sizze_12745 * num_groups_12765;
        struct memblock_device mem_12960;
        
        mem_12960.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12960, bytes_12955, "mem_12960"))
            return 1;
        
        struct memblock_local mem_12950;
        
        mem_12950.references = NULL;
        
        struct memblock_local mem_12954;
        
        mem_12954.references = NULL;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12386, 0, bytes_12947,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12386, 1, bytes_12947,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12386, 2,
                                                sizeof(res_12052), &res_12052));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12386, 3,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12386, 4,
                                                sizeof(mem_12934.mem),
                                                &mem_12934.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12386, 5,
                                                sizeof(mem_12946.mem),
                                                &mem_12946.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12386, 6,
                                                sizeof(mem_12960.mem),
                                                &mem_12960.mem));
        if (1 * (num_groups_12765 * tiled_group_sizze_12745) != 0) {
            const size_t global_work_sizze_13136[1] = {num_groups_12765 *
                         tiled_group_sizze_12745};
            const size_t local_work_sizze_13140[1] = {tiled_group_sizze_12745};
            int64_t time_start_13137 = 0, time_end_13138 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_12386");
                fprintf(stderr, "%zu", global_work_sizze_13136[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13140[0]);
                fprintf(stderr, "].\n");
                time_start_13137 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->map_12386, 1,
                                                            NULL,
                                                            global_work_sizze_13136,
                                                            local_work_sizze_13140,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13138 = get_wall_time();
                
                long time_diff_13139 = time_end_13138 - time_start_13137;
                
                ctx->map_12386_total_runtime += time_diff_13139;
                ctx->map_12386_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12386",
                        time_diff_13139);
            }
        }
        if (memblock_unref_device(ctx, &mem_12946, "mem_12946") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12950, "mem_12950") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12954, "mem_12954") != 0)
            return 1;
        
        int32_t y_12791 = tile_sizze_12779 - 1;
        int32_t x_12792 = 2 + y_12791;
        int32_t groups_in_dim_12793 = squot32(x_12792, tile_sizze_12779);
        int32_t y_12798 = groups_in_dim_12793 * groups_in_dim_12793;
        int32_t y_12799 = res_12055 * y_12798;
        int32_t num_groups_12800 = res_12052 * y_12799;
        int32_t num_threads_12801 = tiled_group_sizze_12780 * num_groups_12800;
        struct memblock_device mem_12974;
        
        mem_12974.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12974, bytes_12969, "mem_12974"))
            return 1;
        
        struct memblock_local mem_12964;
        
        mem_12964.references = NULL;
        
        struct memblock_local mem_12968;
        
        mem_12968.references = NULL;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12337, 0, bytes_12961,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12337, 1, bytes_12961,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12337, 2,
                                                sizeof(res_12052), &res_12052));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12337, 3,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12337, 4,
                                                sizeof(mem_12934.mem),
                                                &mem_12934.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12337, 5,
                                                sizeof(mem_12960.mem),
                                                &mem_12960.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_12337, 6,
                                                sizeof(mem_12974.mem),
                                                &mem_12974.mem));
        if (1 * (num_groups_12800 * tiled_group_sizze_12780) != 0) {
            const size_t global_work_sizze_13141[1] = {num_groups_12800 *
                         tiled_group_sizze_12780};
            const size_t local_work_sizze_13145[1] = {tiled_group_sizze_12780};
            int64_t time_start_13142 = 0, time_end_13143 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_12337");
                fprintf(stderr, "%zu", global_work_sizze_13141[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13145[0]);
                fprintf(stderr, "].\n");
                time_start_13142 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->map_12337, 1,
                                                            NULL,
                                                            global_work_sizze_13141,
                                                            local_work_sizze_13145,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13143 = get_wall_time();
                
                long time_diff_13144 = time_end_13143 - time_start_13142;
                
                ctx->map_12337_total_runtime += time_diff_13144;
                ctx->map_12337_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_12337",
                        time_diff_13144);
            }
        }
        if (memblock_unref_device(ctx, &mem_12934, "mem_12934") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12960, "mem_12960") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12964, "mem_12964") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12968, "mem_12968") != 0)
            return 1;
        
        struct memblock_device mem_12980;
        
        mem_12980.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12980, bytes_12969, "mem_12980"))
            return 1;
        if (res_12052 * (res_12055 * 2 * 2) * sizeof(float) > 0) {
            OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                         mem_12974.mem,
                                                         mem_12980.mem, 0, 0,
                                                         res_12052 *
                                                         (res_12055 * 2 * 2) *
                                                         sizeof(float), 0, NULL,
                                                         NULL));
            if (ctx->debugging)
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
        }
        if (memblock_unref_device(ctx, &mem_12974, "mem_12974") != 0)
            return 1;
        
        int32_t x_12152 = arg_12060 * arg_12061;
        bool assert_arg_12153 = x_12152 == flat_dim_12063;
        bool dim_ok_12154;
        
        if (!assert_arg_12153) {
            ctx->error = msgprintf("Error at %s:\n%s\n",
                                   "winograd.fut:61:3-74:10 -> winograd.fut:71:9-74 -> winograd.fut:44:8-58 -> /futlib/array.fut:87:3-33",
                                   "new shape has different number of elements than old shape");
            if (memblock_unref_device(ctx, &mem_12980, "mem_12980") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12968, "mem_12968") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12964, "mem_12964") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12974, "mem_12974") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12954, "mem_12954") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12950, "mem_12950") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12960, "mem_12960") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12946, "mem_12946") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12940, "mem_12940") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12934, "mem_12934") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12931, "mem_12931") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12926, "mem_12926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12921, "mem_12921") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12909, "mem_12909") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12905, "mem_12905") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12915, "mem_12915") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12895, "mem_12895") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12891, "mem_12891") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12901, "mem_12901") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12887, "mem_12887") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12881, "mem_12881") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12875, "mem_12875") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12872, "mem_12872") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12867, "mem_12867") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12862, "mem_12862") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12858, "mem_12858") != 0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_12991, "res_mem_12991") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12854, "mem_12854") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12851, "mem_12851") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12845, "mem_12845") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12841, "mem_12841") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12848, "mem_12848") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12837, "mem_12837") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12831, "mem_12831") != 0)
                return 1;
            if (memblock_unref_local(ctx, &mem_12827, "mem_12827") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12834, "mem_12834") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12823, "mem_12823") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_12820, "mem_12820") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_13002, "out_mem_13002") !=
                0)
                return 1;
            return 1;
        }
        
        struct memblock_device mem_12984;
        
        mem_12984.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12984, bytes_12981, "mem_12984"))
            return 1;
        
        int32_t group_sizze_13073;
        
        group_sizze_13073 = ctx->sizes.mainzigroup_sizze_13073;
        
        int32_t num_groups_13074;
        
        num_groups_13074 = squot32(sizze_12064 * sizze_12065 +
                                   sext_i32_i32(group_sizze_13073) - 1,
                                   sext_i32_i32(group_sizze_13073));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_13070, 0,
                                                sizeof(res_12055), &res_12055));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_13070, 1,
                                                sizeof(sizze_12064),
                                                &sizze_12064));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_13070, 2,
                                                sizeof(sizze_12065),
                                                &sizze_12065));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_13070, 3,
                                                sizeof(mem_12980.mem),
                                                &mem_12980.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_13070, 4,
                                                sizeof(mem_12984.mem),
                                                &mem_12984.mem));
        if (1 * (num_groups_13074 * group_sizze_13073) != 0) {
            const size_t global_work_sizze_13146[1] = {num_groups_13074 *
                         group_sizze_13073};
            const size_t local_work_sizze_13150[1] = {group_sizze_13073};
            int64_t time_start_13147 = 0, time_end_13148 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "copy_13070");
                fprintf(stderr, "%zu", global_work_sizze_13146[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13150[0]);
                fprintf(stderr, "].\n");
                time_start_13147 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->copy_13070, 1,
                                                            NULL,
                                                            global_work_sizze_13146,
                                                            local_work_sizze_13150,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13148 = get_wall_time();
                
                long time_diff_13149 = time_end_13148 - time_start_13147;
                
                ctx->copy_13070_total_runtime += time_diff_13149;
                ctx->copy_13070_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "copy_13070",
                        time_diff_13149);
            }
        }
        if (memblock_unref_device(ctx, &mem_12980, "mem_12980") != 0)
            return 1;
        if (memblock_set_device(ctx, &res_mem_12991, &mem_12984, "mem_12984") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12984, "mem_12984") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12980, "mem_12980") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12968, "mem_12968") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12964, "mem_12964") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12974, "mem_12974") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12954, "mem_12954") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12950, "mem_12950") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12960, "mem_12960") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12946, "mem_12946") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12940, "mem_12940") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12934, "mem_12934") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12931, "mem_12931") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12926, "mem_12926") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12921, "mem_12921") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12909, "mem_12909") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12905, "mem_12905") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12915, "mem_12915") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12895, "mem_12895") != 0)
            return 1;
        if (memblock_unref_local(ctx, &mem_12891, "mem_12891") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12901, "mem_12901") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12887, "mem_12887") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12881, "mem_12881") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12875, "mem_12875") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12872, "mem_12872") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12867, "mem_12867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12862, "mem_12862") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12858, "mem_12858") != 0)
            return 1;
    } else {
        struct memblock_device mem_12989;
        
        mem_12989.references = NULL;
        if (memblock_alloc_device(ctx, &mem_12989, bytes_12981, "mem_12989"))
            return 1;
        
        int32_t group_sizze_13078;
        
        group_sizze_13078 = ctx->sizes.mainzigroup_sizze_13078;
        
        int32_t num_groups_13079;
        
        num_groups_13079 = squot32(sizze_12064 * sizze_12065 +
                                   sext_i32_i32(group_sizze_13078) - 1,
                                   sext_i32_i32(group_sizze_13078));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_13075, 0,
                                                sizeof(sizze_12064),
                                                &sizze_12064));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_13075, 1,
                                                sizeof(sizze_12065),
                                                &sizze_12065));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_13075, 2,
                                                sizeof(mem_12989.mem),
                                                &mem_12989.mem));
        if (1 * (num_groups_13079 * group_sizze_13078) != 0) {
            const size_t global_work_sizze_13151[1] = {num_groups_13079 *
                         group_sizze_13078};
            const size_t local_work_sizze_13155[1] = {group_sizze_13078};
            int64_t time_start_13152 = 0, time_end_13153 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "replicate_13075");
                fprintf(stderr, "%zu", global_work_sizze_13151[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_13155[0]);
                fprintf(stderr, "].\n");
                time_start_13152 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->replicate_13075,
                                                            1, NULL,
                                                            global_work_sizze_13151,
                                                            local_work_sizze_13155,
                                                            0, NULL, NULL));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_13153 = get_wall_time();
                
                long time_diff_13154 = time_end_13153 - time_start_13152;
                
                ctx->replicate_13075_total_runtime += time_diff_13154;
                ctx->replicate_13075_runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "replicate_13075",
                        time_diff_13154);
            }
        }
        if (memblock_set_device(ctx, &res_mem_12991, &mem_12989, "mem_12989") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mem_12989, "mem_12989") != 0)
            return 1;
    }
    if (memblock_unref_device(ctx, &mem_12820, "mem_12820") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12848, "mem_12848") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12851, "mem_12851") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12854, "mem_12854") != 0)
        return 1;
    out_arrsizze_13004 = sizze_12064;
    out_arrsizze_13005 = sizze_12065;
    out_memsizze_13003 = bytes_12981;
    if (memblock_set_device(ctx, &out_mem_13002, &res_mem_12991,
                            "res_mem_12991") != 0)
        return 1;
    *out_out_memsizze_13080 = out_memsizze_13003;
    (*out_mem_p_13081).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_13081, &out_mem_13002,
                            "out_mem_13002") != 0)
        return 1;
    *out_out_arrsizze_13082 = out_arrsizze_13004;
    *out_out_arrsizze_13083 = out_arrsizze_13005;
    if (memblock_unref_device(ctx, &res_mem_12991, "res_mem_12991") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12854, "mem_12854") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12851, "mem_12851") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_12845, "mem_12845") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_12841, "mem_12841") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12848, "mem_12848") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12837, "mem_12837") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_12831, "mem_12831") != 0)
        return 1;
    if (memblock_unref_local(ctx, &mem_12827, "mem_12827") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12834, "mem_12834") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12823, "mem_12823") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_12820, "mem_12820") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_13002, "out_mem_13002") != 0)
        return 1;
    return 0;
}
static int futrts__map_transpose_f32(struct futhark_context *ctx,
                                     struct memblock_device destmem_0,
                                     int32_t destoffset_1,
                                     struct memblock_device srcmem_2,
                                     int32_t srcoffset_3, int32_t num_arrays_4,
                                     int32_t x_elems_5, int32_t y_elems_6,
                                     int32_t in_elems_7, int32_t out_elems_8)
{
    if (!(num_arrays_4 == 0 || (x_elems_5 == 0 || y_elems_6 == 0))) {
        int32_t muly_10 = squot32(16, x_elems_5);
        int32_t mulx_9 = squot32(16, y_elems_6);
        
        if (in_elems_7 == out_elems_8 && ((num_arrays_4 == 1 || x_elems_5 *
                                           y_elems_6 == in_elems_7) &&
                                          (x_elems_5 == 1 || y_elems_6 == 1))) {
            if (in_elems_7 * sizeof(float) > 0) {
                OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                             srcmem_2.mem,
                                                             destmem_0.mem,
                                                             srcoffset_3,
                                                             destoffset_1,
                                                             in_elems_7 *
                                                             sizeof(float), 0,
                                                             NULL, NULL));
                if (ctx->debugging)
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            }
        } else {
            if (sle32(x_elems_5, 8) && slt32(16, y_elems_6)) {
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        0, sizeof(destoffset_1),
                                                        &destoffset_1));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        1, sizeof(srcoffset_3),
                                                        &srcoffset_3));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        2, sizeof(num_arrays_4),
                                                        &num_arrays_4));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        3, sizeof(x_elems_5),
                                                        &x_elems_5));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        4, sizeof(y_elems_6),
                                                        &y_elems_6));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        5, sizeof(in_elems_7),
                                                        &in_elems_7));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        6, sizeof(out_elems_8),
                                                        &out_elems_8));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        7, sizeof(mulx_9),
                                                        &mulx_9));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        8, sizeof(muly_10),
                                                        &muly_10));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        9,
                                                        sizeof(destmem_0.mem),
                                                        &destmem_0.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        10,
                                                        sizeof(srcmem_2.mem),
                                                        &srcmem_2.mem));
                if (1 * (squot32(x_elems_5 + 16 - 1, 16) * 16) *
                    (squot32(squot32(y_elems_6 + muly_10 - 1, muly_10) + 16 - 1,
                             16) * 16) * (num_arrays_4 * 1) != 0) {
                    const size_t global_work_sizze_13156[3] =
                                 {squot32(x_elems_5 + 16 - 1, 16) * 16,
                                  squot32(squot32(y_elems_6 + muly_10 - 1,
                                                  muly_10) + 16 - 1, 16) * 16,
                                  num_arrays_4 * 1};
                    const size_t local_work_sizze_13160[3] = {16, 16, 1};
                    int64_t time_start_13157 = 0, time_end_13158 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "map_transpose_f32_low_width");
                        fprintf(stderr, "%zu", global_work_sizze_13156[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_13156[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_13156[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_13160[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_13160[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_13160[2]);
                        fprintf(stderr, "].\n");
                        time_start_13157 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->map_transpose_f32_low_width,
                                                                    3, NULL,
                                                                    global_work_sizze_13156,
                                                                    local_work_sizze_13160,
                                                                    0, NULL,
                                                                    NULL));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_13158 = get_wall_time();
                        
                        long time_diff_13159 = time_end_13158 -
                             time_start_13157;
                        
                        ctx->map_transpose_f32_low_width_total_runtime +=
                            time_diff_13159;
                        ctx->map_transpose_f32_low_width_runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "map_transpose_f32_low_width", time_diff_13159);
                    }
                }
            } else {
                if (sle32(y_elems_6, 8) && slt32(16, x_elems_5)) {
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            0,
                                                            sizeof(destoffset_1),
                                                            &destoffset_1));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            1,
                                                            sizeof(srcoffset_3),
                                                            &srcoffset_3));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            2,
                                                            sizeof(num_arrays_4),
                                                            &num_arrays_4));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            3,
                                                            sizeof(x_elems_5),
                                                            &x_elems_5));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            4,
                                                            sizeof(y_elems_6),
                                                            &y_elems_6));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            5,
                                                            sizeof(in_elems_7),
                                                            &in_elems_7));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            6,
                                                            sizeof(out_elems_8),
                                                            &out_elems_8));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            7, sizeof(mulx_9),
                                                            &mulx_9));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            8, sizeof(muly_10),
                                                            &muly_10));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            9,
                                                            sizeof(destmem_0.mem),
                                                            &destmem_0.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            10,
                                                            sizeof(srcmem_2.mem),
                                                            &srcmem_2.mem));
                    if (1 * (squot32(squot32(x_elems_5 + mulx_9 - 1, mulx_9) +
                                     16 - 1, 16) * 16) * (squot32(y_elems_6 +
                                                                  16 - 1, 16) *
                                                          16) * (num_arrays_4 *
                                                                 1) != 0) {
                        const size_t global_work_sizze_13161[3] =
                                     {squot32(squot32(x_elems_5 + mulx_9 - 1,
                                                      mulx_9) + 16 - 1, 16) *
                                      16, squot32(y_elems_6 + 16 - 1, 16) * 16,
                                      num_arrays_4 * 1};
                        const size_t local_work_sizze_13165[3] = {16, 16, 1};
                        int64_t time_start_13162 = 0, time_end_13163 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "map_transpose_f32_low_height");
                            fprintf(stderr, "%zu", global_work_sizze_13161[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_13161[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_13161[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_13165[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_13165[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_13165[2]);
                            fprintf(stderr, "].\n");
                            time_start_13162 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->map_transpose_f32_low_height,
                                                                        3, NULL,
                                                                        global_work_sizze_13161,
                                                                        local_work_sizze_13165,
                                                                        0, NULL,
                                                                        NULL));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_13163 = get_wall_time();
                            
                            long time_diff_13164 = time_end_13163 -
                                 time_start_13162;
                            
                            ctx->map_transpose_f32_low_height_total_runtime +=
                                time_diff_13164;
                            ctx->map_transpose_f32_low_height_runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "map_transpose_f32_low_height",
                                    time_diff_13164);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, 8) && sle32(y_elems_6, 8)) {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                0,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                1,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                2,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                3,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                4,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                5,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                6,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                7,
                                                                sizeof(mulx_9),
                                                                &mulx_9));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                8,
                                                                sizeof(muly_10),
                                                                &muly_10));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                9,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                10,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * (squot32(num_arrays_4 * x_elems_5 * y_elems_6 +
                                         256 - 1, 256) * 256) != 0) {
                            const size_t global_work_sizze_13166[1] =
                                         {squot32(num_arrays_4 * x_elems_5 *
                                                  y_elems_6 + 256 - 1, 256) *
                                         256};
                            const size_t local_work_sizze_13170[1] = {256};
                            int64_t time_start_13167 = 0, time_end_13168 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "map_transpose_f32_small");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_13166[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_13170[0]);
                                fprintf(stderr, "].\n");
                                time_start_13167 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->map_transpose_f32_small,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_13166,
                                                                            local_work_sizze_13170,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_13168 = get_wall_time();
                                
                                long time_diff_13169 = time_end_13168 -
                                     time_start_13167;
                                
                                ctx->map_transpose_f32_small_total_runtime +=
                                    time_diff_13169;
                                ctx->map_transpose_f32_small_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "map_transpose_f32_small",
                                        time_diff_13169);
                            }
                        }
                    } else {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                0,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                1,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                2,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                3,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                4,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                5,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                6,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                7,
                                                                sizeof(mulx_9),
                                                                &mulx_9));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                8,
                                                                sizeof(muly_10),
                                                                &muly_10));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                9,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                10,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * (squot32(x_elems_5 + 32 - 1, 32) * 32) *
                            (squot32(y_elems_6 + 32 - 1, 32) * 8) *
                            (num_arrays_4 * 1) != 0) {
                            const size_t global_work_sizze_13171[3] =
                                         {squot32(x_elems_5 + 32 - 1, 32) * 32,
                                          squot32(y_elems_6 + 32 - 1, 32) * 8,
                                          num_arrays_4 * 1};
                            const size_t local_work_sizze_13175[3] = {32, 8, 1};
                            int64_t time_start_13172 = 0, time_end_13173 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "map_transpose_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_13171[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_13171[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_13171[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_13175[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_13175[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_13175[2]);
                                fprintf(stderr, "].\n");
                                time_start_13172 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->map_transpose_f32,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_13171,
                                                                            local_work_sizze_13175,
                                                                            0,
                                                                            NULL,
                                                                            NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_13173 = get_wall_time();
                                
                                long time_diff_13174 = time_end_13173 -
                                     time_start_13172;
                                
                                ctx->map_transpose_f32_total_runtime +=
                                    time_diff_13174;
                                ctx->map_transpose_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "map_transpose_f32", time_diff_13174);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
struct futhark_f32_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr = malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(float),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (dim0 * dim1 * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * dim1 *
                                                      sizeof(float), data + 0,
                                                      0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              cl_mem data, int offset, int dim0,
                                              int dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr = malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(float),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (dim0 * dim1 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * dim1 *
                                                     sizeof(float), 0, NULL,
                                                     NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * arr->shape[1] * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     arr->shape[1] *
                                                     sizeof(float), data + 0, 0,
                                                     NULL, NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                 struct futhark_f32_2d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr)
{
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0, const
                       struct futhark_f32_2d *in0, const
                       struct futhark_f32_2d *in1)
{
    int64_t image_mem_sizze_12813;
    struct memblock_device image_mem_12814;
    
    image_mem_12814.references = NULL;
    
    int64_t kernel_mem_sizze_12815;
    struct memblock_device kernel_mem_12816;
    
    kernel_mem_12816.references = NULL;
    
    int32_t sizze_11973;
    int32_t sizze_11974;
    int32_t sizze_11975;
    int32_t sizze_11976;
    int64_t out_memsizze_13003;
    struct memblock_device out_mem_13002;
    
    out_mem_13002.references = NULL;
    
    int32_t out_arrsizze_13004;
    int32_t out_arrsizze_13005;
    
    lock_lock(&ctx->lock);
    image_mem_12814 = in0->mem;
    image_mem_sizze_12813 = in0->mem.size;
    sizze_11973 = in0->shape[0];
    sizze_11974 = in0->shape[1];
    kernel_mem_12816 = in1->mem;
    kernel_mem_sizze_12815 = in1->mem.size;
    sizze_11975 = in1->shape[0];
    sizze_11976 = in1->shape[1];
    
    int ret = futrts_main(ctx, &out_memsizze_13003, &out_mem_13002,
                          &out_arrsizze_13004, &out_arrsizze_13005,
                          image_mem_sizze_12813, image_mem_12814,
                          kernel_mem_sizze_12815, kernel_mem_12816, sizze_11973,
                          sizze_11974, sizze_11975, sizze_11976);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_13002;
        (*out0)->shape[0] = out_arrsizze_13004;
        (*out0)->shape[1] = out_arrsizze_13005;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
