#include "hmm.h"
#define MODELS 5

HMM hmm;

// char modellist_path[100] = "modellist.txt";
// char test_path[100] = "testing_data1.txt";
// char result_path[100] = "result1.txt";

int main(int argc, char *argv[]){
	//parameters
	strcpy(modellist_path, argv[1]);
	strcpy(test_path, argv[2]);
	strcpy(result_path, argv[3]);

	//input
	int row, col;
	char test_data[MAX_ROW][MAX_COL];
	read_train_data(test_data, test_path, &row, &col);

	FILE *fp = open_or_die(modellist_path, "r");

	char token[20];
	HMM hmm[MODELS];

	int i=0;
	while(fscanf(fp, "%s", token) > 0){
		if(token[0] == '\0' || token[0] == '\n') continue;
		loadHMM(&hmm[i], token);
		log_norm(&hmm[i]);
		i++;
	}

	int results[MAX_ROW];
	for(int i=0; i<row; i++){
		double mx_val = -100000000000;
		int res;
		for(int j=0; j<MODELS; j++){
			double val = viterbi(&hmm[j], test_data[i], col);
			if (mx_val < val){
				res = j + 1;
				mx_val = val;
			}
		}
		results[i] = res;
	}

	//output results
	FILE *fp_res = open_or_die(result_path, "w");
	for(int i=0; i<row; i++){
		char str[20] = "model_0";
		char buffer[2];
		sprintf(buffer, "%d", results[i]);
		strcat(str, buffer);
		strcat(str, ".txt");

		fprintf(fp_res, "%s\n", str);
	}

	return 0;	
}







