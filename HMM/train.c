#include "hmm.h"
HMM hmm;

// //parameters
// int iteration = 5;
// char init_path[100] = "model_init.txt";
// char train_path[100] = "seq_model_01.txt";
// char model_path[100] = "model_01.txt";

int main(int argc, char *argv[]){
	//parameters
	iteration = atoi(argv[1]);
	strcpy(init_path, argv[2]);
	strcpy(train_path, argv[3]);
	strcpy(model_path, argv[4]);

	//input
	loadHMM( &hmm, init_path);
	log_norm(&hmm);

	char train_data[MAX_ROW][MAX_COL];
	int row, col;
	read_train_data(train_data, train_path, &row, &col);

	for(int i=0; i<iteration; i++)
		EM_train(&hmm, train_data, row, col);


	FILE *fp = open_or_die(model_path, "w");

	log_scale(&hmm);
	dumpHMM( fp, &hmm );

	printf("%s Done!\n", model_path);
	return 0;	
}





































