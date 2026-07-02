// SoA dictionary
// Simple dictionary implementation using open addresing, linear probing
// the input values are not hashed since we assume that keys have been
// already hashed (context: long message attack)

#include "confg_math_func.h"
#include "numbers_shorthands.h"
#include "dict.h"
#include "config.h"
//#include "util_char_arrays.h"
#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "shared.h"
#include "util_char_arrays.h"
#include <sys/mman.h> 
#include <immintrin.h>

//  how many bits we store as a value in dictionary


/* void print_m256i(__m256i a, char* text){ */
/*   uint32_t A[8] = {0}; */
/*   _mm256_storeu_si256((__m256i*)A, a); */
/*   printf("%s = ", text); */
/*   for (int i = 0; i<8; ++i) { */
/*     printf("%02x, ", A[i]); */
/*   } */
/*   puts(""); */
/* } */



/* //-----------------------------------------------------// */
/* //                  data structure                     // */
/* //-----------------------------------------------------// */
/// See the header file

//-----------------------------------------------------//
//                       methods                       //
//-----------------------------------------------------//
/* represent n in <= 6 char  */

dict *dict_new(size_t nelements) {
  /// dict v5: simple bounded linear probing. avoid unnecessary complications.
  /// dict v4: we take part of the value as in index and store the rest
  ///          as a value. The value is maximally 32 bits 
  /// dict v3: we don't store values, we only store 64bit of the key
  ///         in d->keys[k]. When d->keys[k]=0, it means that we have
  ///         we have an empty slot


  dict* d = (dict *) aligned_alloc(ALIGNMENT, (sizeof(dict)));

  /* int nslots_per_bucket = SIMD_LEN; /\* See ../include/config.h *\/ */
  /* size_t nslots = nelements; */

  //+ todo does negative gives the correct result here?
  //nslots = nslots + (-nslots % nslots_per_bucket);


  /// Save configured variables as dictionary entries 

  d->nslots = nelements;

  /* /\* for huge pages : required memory for nslots is a mutlipe GPAGE_SIZE  *\/ */
  /* d->nslots = ((d->nslots*sizeof(VAL_TYPE)) / GPAGE_SIZE ) * GPAGE_SIZE; */
  /* d->nslots = d->nslots / sizeof(VAL_TYPE); */
    
  d->nelements = 0; /* how many elements currently in the dictionary */
  d->nelements_asked_to_be_inserted = 0;

  d->nprobes_insert=0;
  d->nprobes_lookup=0;

  // the extra d->nslots_per_bucket seems to supress the address sanitizer
  // error, however, i am not sure why since all accesses are < nslots.
  /* d->values = (VAL_TYPE*) aligned_alloc(ALIGNMENT, */
  /* 				      (nslots+d->nslots_per_bucket)*(sizeof(VAL_TYPE))); */
  d->values = (VAL_TYPE*) aligned_alloc(GPAGE_SIZE,
					(d->nslots)*(sizeof(VAL_TYPE))
					+ GPAGE_SIZE);
             /* address sanitizer complains wihtout the above addition */

  madvise(d->values,
	  GPAGE_SIZE,
	  MADV_HUGEPAGE);
  
  /* d->values = (VAL_TYPE*) malloc((nslots)*(sizeof(VAL_TYPE))); */

  // Ensure keys are zeroed
  //#pragma omp simd
  for (size_t i = 0; i < (d->nslots); ++i) 
     d->values[i] = 0; // 0 if it is not occupied

  return d;
}

inline void dict_free(dict *d) { free(d->values); }

size_t dict_memory(size_t nelements){
  size_t estimate = nelements*(sizeof(VAL_TYPE)) + sizeof(dict) + GPAGE_SIZE;
  
  return estimate;
}




int dict_add_element_to(dict* d, u8* state){
  // =========================================================================+
  // returns 1 if an element has been added, 0 otherwise                      |
  // This dictionary is unusual:                                              |
  // User have a value in the form:                                           |
  // (dist pts, srvr no) || (L bits) || discard || (VAL_SIZE bits) || discard |
  // Dictionary expects user to pass: (L bits) || discard || (VAL_SIZE bits)  |
  // ------------------------------------------------------------------------ |
  // we don't store (dist pts, server) since they are already determined      |
  // The discarded bits between L and VAL_SIZE are due to the fact we move 1  |
  // at least. Those we choose to forget them. The last disarded bits are     |
  // discarded because they will double the dictionary size if we include em  |
  // -------------------------------------------------------------------------|
  // INPUTS:                                                                  |
  // `*d`:  dictionary that will store state as an element                    |
  // `*state`: element to be stored in *d in the form                         |
  //          (L bits) || discard || (VAL_SIZE bits) more precisely:          |
  //          (L_IN_BYTES bytes)  || (VAL_SIZE bits)                          |
  // issues may aris when VAL_SIZE is larger then what is left in the state   |
  //                                                                          |
  // -------------------------------------------------------------------------+
  /* how many bytes do we need to index the buckets */
  const int idx_size = (int)ceil(log2(NSLOTS_MY_NODE)/8.0) ;

  /* if (VAL_SIZE_BYTES + idx_size > N){ */
  /*   printf("ERROR at adding to dict! since VAL_SIZE_BYTES=%u, idx_size=%d, while N=%u\n", */
  /* 	   VAL_SIZE_BYTES, */
  /* 	   idx_size, */
  /* 	   N); */
  /* } */
  
  /* ++(d->nelements_asked_to_be_inserted); */


  /// Use linear probing to add element to the array d->values
  // get bucket number, recall keys[nbuckets*nslots_per_bucket
  u64 idx = 0;
  memcpy(&idx, state, idx_size);
  /* printf("initially idx=0x%llx, idx_size=%d\n", idx, idx_size); */

  /* get the bucket number and scale the index */
  idx = idx % d->nslots;

  
  VAL_TYPE val = 0;

  memcpy(&val,
	 &state[idx_size],
	 VAL_SIZE_BYTES );

  /* 0 means empty, we have to ignore zero values  */
  if (val == 0) return 0;
  
  // linear probing 
  for (int i=0; i < NPROBES_MAX;  ++i) {
    // found an empty slot inside a bucket
    /* printf("idx=%llu, d->nslots=%lu\n", idx, d->nslots); */
    if (d->values[idx] == 0) { // found an empty slot
      d->values[idx] = val;
      ++(d->nelements); /* successfully added an element */
      return 1;
    }

    if (d->values[idx] == val) { /* repeated element */
      /* not a new element */
      --(d->nelements_asked_to_be_inserted);
      return 1;
    }
    
    ++idx;
    // reduce mod n->slots //
    if (idx >= d->nslots) /* we forgot the equal sign here */
      idx = 0;
  }
  return 0; // element has been added
}





int dict_has_elm(dict *d, u8 *state)
{ // returns 1 if state is found in d, 0 otherwise                            |
  // This dictionary is unusual:                                              |
  // User have a value in the form:                                           |
  // (dist pts, srvr no) || (L bits) || discard || (VAL_SIZE bits) || discard |
  // Dictionary expects user to pass: (L bits) || discard || (VAL_SIZE bits)  |
  // ------------------------------------------------------------------------ |
  // we don't store (dist pts, server) since they are already determined      |
  // The discarded bits between L and VAL_SIZE are due to the fact we move 1  |
  // at least. Those we choose to forget them. The last disarded bits are     |
  // discarded because they will double the dictionary size if we include em  |
  // -------------------------------------------------------------------------|
  // INPUTS:                                                                  |
  // `*d`:  dictionary that will store state as an element                    |
  // `*state`: element to be looked up  in *d in the form                     |
  //          (L bits) || discard || (VAL_SIZE bits)                          |
  // -------------------------------------------------------------------------+
  /* how many bytes do we need to index the buckets */
  const int idx_size =  (int) ceil(log2(NSLOTS_MY_NODE) / 8.0) ;

  u64 idx = 0;
  memcpy(&idx, state, idx_size);

  idx = idx % d->nslots;

  VAL_TYPE val = 0;
  memcpy(&val,
	 &state[idx_size],
	 VAL_SIZE_BYTES );

  /* 0 means empty, we have to ignore zero values  */
  if (val == 0) return 0;

  
  // loop at most NPROBES_MAX/SIMD_LEN since we load SIMD_LEN
  // elements from dictionary each loop.
  for (size_t i=0; i< (int) NPROBES_MAX; ++i) {
    if (d->values[idx] == val)
      return 1; /* we will hash the whole message again */

    ++idx; 
    if (idx >= d->nslots)
      idx = 0;

    
    #ifdef NPROBES_COUNT
    ++(d->nprobes_lookup);
    #endif
  }
  return 0; // no element is found
}



void dict_print(dict* d){
 
  for (size_t b=0; b < (d->nslots); ++b) {
    printf("slot=%lu, "
	   "key = 0x%016x\n",
	   b,
	   d->values[b]);
  }
}
 
