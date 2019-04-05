#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <math.h>
#include <string.h>
#include "Ngram.h"
using namespace std;


int ngram_order = 2;
Vocab voc;
Ngram lm( voc, ngram_order );

double last_poss[2000];
int path[200];

void split(const string& src, const string& delim, vector<string>& dest);
void read_map(string &map_path, map<string, vector<string> > &zhu_big5);
void trans(vector<string> &row, string &res, map<string, vector<string> > &zhu_big5, vector<vector<int> > &Phi);
double getBigramProb(const char *w1, const char *w2);
void viterbi(vector<string> &row, map<string, vector<string> > &zhu_big5, vector<vector<int> > &Phi);

int main(int argc, char* argv[]){
	// Parameters
	string map_path = "env/ZhuYin-Big5.map";
	string test_ipt = "env/1_seg.txt";
	string filename = "env/bigram.lm";
	string order = "2";
	char lm_filename[100];

	for (int i=1; i< argc-1; i++){
		string arg_opt = argv[i];
		string arg = argv[i+1];

		if (arg_opt == "-text"){
			test_ipt = arg;
		}
		else if (arg_opt == "-map"){
			map_path = arg;
		}
		else if (arg_opt == "-lm"){
			filename = arg;
		}
		else if (arg_opt == "-order"){
			order = arg;
		}
	}


	{
        strcpy(lm_filename, filename.c_str());
        File lmFile( lm_filename, "r" );
        lm.read(lmFile);
        lmFile.close();
    }

	map<string, vector<string> > zhu_big5;
	vector<vector<int> > Phi;

	read_map(map_path, zhu_big5);
	
	FILE *ip = fopen(test_ipt.c_str(), "r");

	vector<string> row;
	char Big5[3];
	char space;
	string res;

	char line[500];
	string delim = " ";

	int r = 0;
	while (fgets(line, 500, ip)){
		split(line, delim, row);
		trans(row, res, zhu_big5, Phi);
		cout << res;
		row.clear();
	}

	return 0;
}


void read_map(string &map_path, map<string, vector<string> > &zhu_big5){
	char space;
	char ZhuYin[3];
	char Big5[3];

	FILE *fp = fopen(map_path.c_str(), "r");

	while(fscanf(fp, "%s%c", ZhuYin, &space) != EOF){
		while(fscanf(fp, "%s%c", Big5, &space) && space != '\n'){
			if (zhu_big5.find(ZhuYin) == zhu_big5.end()){
				vector<string> big5s;
				big5s.push_back(Big5);
				zhu_big5[ZhuYin] = big5s;
			}
			else{
				zhu_big5[ZhuYin].push_back(Big5);
			}
		}
		zhu_big5[ZhuYin].push_back(Big5);
	}
	fclose(fp);
}

void trans(vector<string> &row, string &res, map<string, vector<string> > &zhu_big5, vector<vector<int> > &Phi){
	viterbi(row, zhu_big5, Phi);

	//get res from path
	res = "<s> ";
	int i;
	for(i=0; i< row.size(); i++){
		if (path[i] == -1)
			res = res + row[i] + " ";
		else{
			string zhu = row[i];
			res = res + zhu_big5[zhu][path[i]] + " ";
		}
	}

	res += "</s>\n";
}

double getBigramProb(const char *w1, const char *w2)
{
    VocabIndex wid1 = voc.getIndex(w1);
    VocabIndex wid2 = voc.getIndex(w2);

    if(wid1 == Vocab_None)  //OOV
        wid1 = voc.getIndex(Vocab_Unknown);
    if(wid2 == Vocab_None)  //OOV
        wid2 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = { wid1, Vocab_None };
    return lm.wordProb( wid2, context);
}

void viterbi(vector<string> &row, map<string, vector<string> > &zhu_big5, vector<vector<int> > &Phi){
	int l = row.size();
	Phi.clear();
	
	double last_len = 0;
	double max;
	double poss;
	int max_idx;
	int last_big_state;

	// First word
	string word = row[0];
	// case 1, word is Big5
	if(zhu_big5.find(word) == zhu_big5.end()){
		// get max possibility
		poss = getBigramProb(string("<s>").c_str(), word.c_str());
		last_poss[0] = poss;
		
		last_big_state = 1;
		last_len = 1;
	}
	// case 2, word is zhuyin
	else{
		max = -100000000;
		for(int i=0; i<zhu_big5[word].size(); i++){
			poss = getBigramProb(string("<s>").c_str(), zhu_big5[word][i].c_str());
			last_poss[i] = poss;
		}

		last_big_state = 0;
		last_len = zhu_big5[word].size();
	}

	// cur update:
	// 	   last_len, last_big_state, last_poss
	// last updata:
	// 	   Phi
	for(int i=1; i<l; i++){
		string word = row[i];
		string last_word = row[i-1];
		// case 1, word is Big5
		if(zhu_big5.find(word) == zhu_big5.end()){
			// get max possibility on last state
			max = -1000000000;
			// case 1: last word is Big5
			if (last_big_state == 1){
				poss = getBigramProb(last_word.c_str(), word.c_str()) + last_poss[0];

				last_poss[0] = poss;
				vector<int > v(1, -1);
				Phi.push_back(v);
			}
			// case 2: last word is zhuyin
			else{
				for(int j=0; j<last_len; j++){
					poss = getBigramProb(zhu_big5[last_word][j].c_str(), word.c_str()) + last_poss[j];
					if (max < poss){
						max_idx = j;
						max = poss;
					}
				}

				last_poss[0] = max;
				vector<int > v(1, max_idx);
				Phi.push_back(v);
			}

			last_len = 1;
			last_big_state = 1;
		}
		// case 2, word is zhuyin
		else{
			// get max possibility on last state
			max = -1000000000;
			// case 1: last word is Big5
			if (last_big_state == 1){
				vector<int > v;

				for (int j = 0; j < zhu_big5[word].size(); j++){
					poss = getBigramProb(last_word.c_str(), zhu_big5[word][j].c_str()) + last_poss[0];
					
					last_poss[j] = poss;
					v.push_back(-1);
				}

				Phi.push_back(v);
			}
			// case 2: last word is zhuyin
			else{
				vector<int > v;

				for(int j=0; j< zhu_big5[word].size(); j++){
					max = -1000000000;
					for(int k=0; k < last_len; k++){
						poss = getBigramProb(zhu_big5[last_word][k].c_str(), zhu_big5[word][j].c_str()) + last_poss[k];	
						if (poss > max){
							max = poss;
							max_idx = k;
						}
					}

					last_poss[j] = poss;
					v.push_back(max_idx);
				}

				Phi.push_back(v);
			}

			last_len = zhu_big5[word].size();
			last_big_state = 0;
		}
	}


	// find maximum last word
	max = -100000000;
	if (last_big_state == 1)
		max_idx = -1;
	else{
		for(int i=0; i<last_len; i++){
			if(last_poss[i] > max) {
				max = last_poss[i];
				max_idx = i;
			}	
		}	
	}

	// get path
	path[l-1] = max_idx;
	for(int i=l-2; i >= 0; i--){
		if (max_idx == -1)
			max_idx = Phi[i][0];
		else
			max_idx = Phi[i][max_idx];
		path[i] = max_idx;
	}

}

void split(const string& src, const string& delim, vector<string>& dest)  
{  
    string str = src;  
    string::size_type start, index;  
    string substr;  
  	
  	start = str.find_first_not_of(delim, 0);
    index = str.find_first_of(delim, start);    //在str中查找(起始：start) delim的任意字符的第一次出现的位置  
    while(index != string::npos)  
    {  
        substr = str.substr(start, index-start);  
        dest.push_back(substr);  
        start = str.find_first_not_of(delim, index);    //在str中查找(起始：index) 第一个不属于delim的字符出现的位置  
        if(start == string::npos) return;  
  
        index = str.find_first_of(delim, start);  
    }  
} 










