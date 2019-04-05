#ifndef HMM_HEADER_
#define HMM_HEADER_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef MAX_STATE
#	define MAX_STATE	10
#endif

#ifndef MAX_OBSERV
#	define MAX_OBSERV	26
#endif

#ifndef MAX_SEQ
#	define	MAX_SEQ		200
#endif

#ifndef MAX_LINE
#	define MAX_LINE 	256
#endif

#define MAX_ROW 10000
#define MAX_COL 50

#define LZERO  (-1.0E10) // log(0)
#define LSMALL (-0.5E10) // log values < LSMALL are set to LZERO
#define minLogExp -log(-LZERO) // ~=-23



typedef struct{
   char *model_name;
   int state_num;             //number of state
   int observ_num;               //number of observation
   double initial[MAX_STATE];       //initial prob.
   double transition[MAX_STATE][MAX_STATE];  //transition prob.
   double observation[MAX_OBSERV][MAX_STATE];   //observation prob.
} HMM;

FILE *open_or_die( const char *filename, const char *ht );
void loadHMM( HMM *hmm, const char *filename );
void dumpHMM( FILE *fp, HMM *hmm );
void read_train_data(char (*train_data)[MAX_COL], char *train_path, int *row, int *col);
void log_norm(HMM *hmm);
void log_scale(HMM *hmm);
double forward(HMM *hmm, char * o, int T);
double backward(HMM *hmm, char * o, int T);
double viterbi(HMM *hmm, char * o, int T);
void EM_train(HMM *hmm, char (* data)[MAX_COL], int row, int T);

#endif





























