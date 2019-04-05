#include "hmm.h"

double LogAdd(double x, double y)
{
    double temp, diff, z;
    if (x < y)
    {
        temp = x; x = y; y = temp;
    }
    diff = y-x; // notice that diff <= 0
    if (diff < minLogExp)   // if y' is far smaller than x'
        return (x < LSMALL) ? LZERO : x;
    else
    {
        z = exp(diff);
        return x + log(1.0 + z);
    }
}

FILE *open_or_die( const char *filename, const char *ht )
{
   FILE *fp = fopen( filename, ht );
   if( fp == NULL ){
      perror( filename);
      exit(1);
   }

   return fp;
}

void loadHMM( HMM *hmm, const char *filename )
{
   int i, j;
   FILE *fp = open_or_die( filename, "r");

   hmm->model_name = (char *)malloc( sizeof(char) * (strlen( filename)+1));
   strcpy( hmm->model_name, filename );

   char token[MAX_LINE] = "";
   while( fscanf( fp, "%s", token ) > 0 )
   {
      if( token[0] == '\0' || token[0] == '\n' ) continue;

      if( strcmp( token, "initial:" ) == 0 ){
         fscanf(fp, "%d", &hmm->state_num );

         for( i = 0 ; i < hmm->state_num ; i++ )
            fscanf(fp, "%lf", &( hmm->initial[i] ) );
      }
      else if( strcmp( token, "transition:" ) == 0 ){
         fscanf(fp, "%d", &hmm->state_num );

         for( i = 0 ; i < hmm->state_num ; i++ )
            for( j = 0 ; j < hmm->state_num ; j++ )
               fscanf(fp, "%lf", &( hmm->transition[i][j] ));
      }
      else if( strcmp( token, "observation:" ) == 0 ){
         fscanf(fp, "%d", &hmm->observ_num );

         for( i = 0 ; i < hmm->observ_num ; i++ )
            for( j = 0 ; j < hmm->state_num ; j++ )
               fscanf(fp, "%lf", &( hmm->observation[i][j]) );
      }
   }
}

void dumpHMM( FILE *fp, HMM *hmm )
{
   int i, j;

   //fprintf( fp, "model name: %s\n", hmm->model_name );
   fprintf( fp, "initial: %d\n", hmm->state_num );
   for( i = 0 ; i < hmm->state_num - 1; i++ )
      fprintf( fp, "%.5lf ", hmm->initial[i]);
   fprintf(fp, "%.5lf\n", hmm->initial[ hmm->state_num - 1 ] );

   fprintf( fp, "\ntransition: %d\n", hmm->state_num );
   for( i = 0 ; i < hmm->state_num ; i++ ){
      for( j = 0 ; j < hmm->state_num - 1 ; j++ )
         fprintf( fp, "%.5lf ", hmm->transition[i][j] );
      fprintf(fp,"%.5lf\n", hmm->transition[i][hmm->state_num - 1]);
   }

   fprintf( fp, "\nobservation: %d\n", hmm->observ_num );
   for( i = 0 ; i < hmm->observ_num ; i++ ){
      for( j = 0 ; j < hmm->state_num - 1 ; j++ )
         fprintf( fp, "%.5lf ", hmm->observation[i][j] );
      fprintf(fp,"%.5lf\n", hmm->observation[i][hmm->state_num - 1]);
   }
}

int load_models( const char *listname, HMM *hmm, const int max_num )
{
   FILE *fp = open_or_die( listname, "r" );

   int count = 0;
   char filename[MAX_LINE] = "";
   while( fscanf(fp, "%s", filename) == 1 ){
      loadHMM( &hmm[count], filename );
      count ++;

      if( count >= max_num ){
         return count;
      }
   }
   fclose(fp);

   return count;
}

void dump_models( HMM *hmm, const int num )
{
   int i = 0;
   for( ; i < num ; i++ ){ 
      //		FILE *fp = open_or_die( hmm[i].model_name, "w" );
      dumpHMM( stderr, &hmm[i] );
   }
}


void read_train_data(char (*train_data)[MAX_COL], char *train_path, int *row, int *col)
{
    FILE *fp = open_or_die(train_path, "r");
   char token[MAX_COL];

   int i=0;
   int j=0;
   while( fscanf( fp, "%s", token ) > 0 )
   {
      if( token[0] == '\0' || token[0] == '\n' ) continue;

      j = 0;
      while(token[j] != '\n' && token[j] != '\0'){
         train_data[i][j] = token[j] - 'A';
         j++;
      }
      i++;

   }
   *row = i;
   *col = j;
}

void log_norm(HMM *hmm){
   for(int i=0; i<hmm->state_num; i++){
      hmm->initial[i] = log(hmm->initial[i]);

      for(int j=0; j<hmm->state_num; j++)
         hmm->transition[i][j] = log(hmm->transition[i][j]);
   }

   for(int i=0; i<hmm->observ_num; i++)
      for(int j=0; j<hmm->state_num; j++)
         hmm->observation[i][j] = log(hmm->observation[i][j]);
}

void log_scale(HMM *hmm){
   for(int i=0; i<hmm->state_num; i++){
      hmm->initial[i] = exp(hmm->initial[i]);

      for(int j=0; j<hmm->state_num; j++)
         hmm->transition[i][j] = exp(hmm->transition[i][j]);
   }

   for(int i=0; i<hmm->observ_num; i++)
      for(int j=0; j<hmm->state_num; j++)
         hmm->observation[i][j] = exp(hmm->observation[i][j]);
}

//hmm model
double alpha[MAX_COL][MAX_STATE], beta[MAX_COL][MAX_STATE];   //evaluation problem
double sigma[MAX_COL][MAX_STATE]; int Phi[MAX_COL][MAX_STATE];// decoding problem
double gamma_[MAX_COL][MAX_STATE], delta[MAX_COL][MAX_STATE][MAX_STATE];  //learning problem

double forward(HMM *hmm, char * o, int T){
   int N = hmm->state_num, M = hmm->observ_num;
   double (*A)[MAX_STATE] = hmm->transition;
   double (*B)[MAX_STATE] = hmm->observation;

   for(int t=0; t < T; t++){
      for(int i=0; i< N; i++){
         if(t == 0)
            alpha[t][i] = hmm->initial[i] + B[o[t]][i];
         else{
            double p = LZERO;
            for(int j=0; j<N; j++)
               p = LogAdd(p, alpha[t-1][j] + A[j][i]);
            alpha[t][i] = p + B[o[t]][i];
         }
      }   
   }

   double p = 0;
   for(int i=0; i<N; i++)
      p = LogAdd(p, alpha[T-1][i]);

   return p;
}

double backward(HMM *hmm, char * o, int T){
   int N = hmm->state_num, M = hmm->observ_num;
   double (*A)[MAX_STATE] = hmm->transition;
   double (*B)[MAX_STATE] = hmm->observation;

   for(int t=T-1; t>=0; t--){
      for(int i=0; i<N; i++){
         if(t == T-1)
            beta[t][i] = log(1.0);
         else{
            double p = LZERO;
            for(int j=0; j<N; j++)
               p = LogAdd(p, A[i][j] + B[o[t+1]][j] + beta[t+1][j]);
            beta[t][i] = p;
         }
      }
   }

   double p = LZERO;
   for(int i=0; i< N; i++)
      p = LogAdd(p, hmm->initial[i] + B[o[0]][i] + beta[0][i]);

   return p;
}

double viterbi(HMM *hmm, char * o, int T){
   int N = hmm->state_num, M = hmm->observ_num;
   double (*A)[MAX_STATE] = hmm->transition;
   double (*B)[MAX_STATE] = hmm->observation;

   for(int t=0; t< T; t++){
      for(int i=0; i<N; i++){
         if(t == 0)
            sigma[t][i] = hmm->initial[i] + B[o[t]][i];
         else{
            double mx = LZERO;
            for(int j=0; j<N; j++){
               double p = sigma[t-1][j] + A[j][i];
               if(p > mx){
                  mx = p;
                  Phi[t][i] = j;
               }
            }
            sigma[t][i] = mx + B[o[t]][i];
         }
      }
   }

   double mx = LZERO;
   char q[T];
   for(int i=0; i<N; i++){
      if(sigma[T-1][i] > mx){
         mx = sigma[T-1][i];
         q[T-1] = i;
      }
   }

   for(int t=T-1; t>=0; t--){
      q[t-1] = Phi[t][q[t]];
   }

   return mx;
}


void EM_train(HMM *hmm, char (* data)[MAX_COL], int row, int T){
   int N = hmm->state_num, M = hmm->observ_num;
   double (*A)[MAX_STATE] = hmm->transition;
   double (*B)[MAX_STATE] = hmm->observation;

   double initial_numer[MAX_STATE] = {0}, transition_numer[MAX_STATE][MAX_STATE] = {{0}}, observation_numer[MAX_OBSERV][MAX_STATE] = {{0}};
   double transition_denom[MAX_STATE] = {0}, observation_denom[MAX_STATE] = {0};

   //initialize
   for(int i=0; i<MAX_STATE; i++){
      initial_numer[i] = LZERO;
      transition_denom[i] = LZERO;
      observation_denom[i] = LZERO;

      for(int j=0; j<MAX_STATE; j++)
         transition_numer[i][j] = LZERO;
   }

   for(int i=0; i<MAX_OBSERV; i++){
      for(int j=0; j<MAX_STATE; j++)
         observation_numer[i][j] = LZERO;
   }


   //updata hmm
   for(int r=0; r<row; r++){
      forward(hmm, data[r], T);
      backward(hmm, data[r], T);

      double p = LZERO;
      for(int i=0; i<N; i++)
         p = LogAdd(p, alpha[0][i] + beta[0][i]);

      for(int t = 0; t<T; t++)
         for(int i=0; i<N; i++)
            gamma_[t][i] = (alpha[t][i] + beta[t][i]) -  p;

      for(int t=0; t<T-1; t++)
         for(int i=0; i<N; i++)
            for(int j=0; j<N; j++)
               delta[t][i][j] = (alpha[t][i] + A[i][j] + B[data[r][t+1]][j] + beta[t+1][j]) - p;

      //accumulate Pi
      for (int i=0; i<N; ++i)
         initial_numer[i] = LogAdd(initial_numer[i], gamma_[0][i]);

      //accumulate A
      for(int i=0; i<N; i++){
         //denominator of state i
         double p2 = LZERO;
         for(int t=0; t<T-1; t++)
            p2 = LogAdd(p2, gamma_[t][i]);

         transition_denom[i] = LogAdd(transition_denom[i], p2);

         for(int j=0; j<N; j++){
            //numerator of A[i][j]
            double p1 = LZERO;
            for(int t=0; t<T-1; t++)
               p1 = LogAdd(p1, delta[t][i][j]);

            transition_numer[i][j] = LogAdd(transition_numer[i][j], p1);
         }
      }

      //accumulate B
      for(int i=0; i<N; i++){
         double p2 = LZERO, p1[MAX_OBSERV];

         for(int m=0; m<MAX_OBSERV; m++)
            p1[m] = LZERO;

         for(int t=0; t<T; t++){
            p2 = LogAdd(p2, gamma_[t][i]);
            p1[data[r][t]] = LogAdd(p1[data[r][t]], gamma_[t][i]);
         }

         for(int m=0; m<M; m++)
            observation_numer[m][i] = LogAdd(observation_numer[m][i], p1[m]);
         observation_denom[i] = LogAdd(observation_denom[i], p2); 
         
      }
      
   }

   //update Pi
   for(int i=0; i<N; i++)
      hmm->initial[i] = initial_numer[i] - log(row);

   //update A
   for(int i=0; i<N; i++)
      for(int j=0; j<N; j++)
         hmm->transition[i][j] = transition_numer[i][j] - transition_denom[i];
      

   //update B
   for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
         hmm->observation[i][j] = observation_numer[i][j] - observation_denom[j];
}