#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "method.h"

#define RKS _RKS_
#define LEN 500

int getJJ(char *file_name){

    int i, count=1;

    FILE *file;

    file = fopen(file_name, "r");

    char text[LEN];

    if(file == NULL) { printf("Not able to open the file."); return 0; }

    while(fgets(text, LEN, file) !=NULL){

    }

    const char *start_str = "jet ";
    const char *end_str = " symbols";

    char *start, *end, result[LEN];

    if (start = strstr(text, start_str)){
        start+= strlen(start_str);
        if (end = strstr(start, end_str)){
            memcpy(result, start, end-start);
        }
    }

    i=0;
    for (i=0; i<strlen(result); i++){
        if (result[i]==',') {
            count++;
        }
    }

    return count;
}

void modifyLastLine(char *sigma_last, char *rho_last, char *line, int sigma, int rho){

    char temp[LEN];

    int symbols = 0;
    int deg = 0;

    // Find the positions of "symbols"
    char* posSymbols = strstr(line, "symbols");

    strncpy(temp, line, posSymbols - line + strlen("symbols"));

    snprintf(sigma_last, LEN+20, "%s %d deg 1;", temp, sigma);
    snprintf(rho_last, LEN+20, "%s %d deg 1;", temp, rho);

}


void createSigmaRho(char *file_name){

    int i, sigma, rho;

    char name[LEN], sigma_name[LEN+20], rho_name[LEN+20];

    char *pos = strchr(file_name, '_');

    memcpy(name, file_name, pos - file_name);

    rho = getJJ(file_name);
    sigma = rho*_RKS_;

    snprintf(sigma_name, sizeof(sigma_name), "%s_%d_1.eq", name, sigma);
    snprintf(rho_name, sizeof(sigma_name), "%s_%d_1.eq", name, rho);

    FILE *file;
    file = fopen(file_name, "r");

    if (strcmp(sigma_name,file_name)!=0 && strcmp(rho_name,file_name)!=0){
        FILE *sigma_file;
        sigma_file = fopen(sigma_name, "w");
    
        FILE *rho_file;
        rho_file = fopen(rho_name, "w");
    
        char line[LEN], prev_line[LEN]="", sigma_last[LEN+20], rho_last[LEN+20];
    
        if(file == NULL) { printf("Not able to open the file."); exit(0); }
    
        while(fgets(line, LEN, file) !=NULL){
    
            if (strcmp(prev_line, "") != 0) {
                fprintf(sigma_file, "%s", prev_line);
                fprintf(rho_file, "%s", prev_line);
            }
    
            strcpy(prev_line, line);
        }
    
        modifyLastLine(sigma_last, rho_last, line, sigma, rho);
        fprintf(sigma_file, "%s", sigma_last);
        fprintf(rho_file, "%s", rho_last);

    } else {
        if (strcmp(sigma_name,file_name)!=0){
            FILE *sigma_file;
            sigma_file = fopen(sigma_name, "w");
        
            char line[LEN], prev_line[LEN]="", sigma_last[LEN+20], rho_last[LEN+20];
        
            if(file == NULL) { printf("Not able to open the file."); exit(0); }
        
            while(fgets(line, LEN, file) !=NULL){
        
                if (strcmp(prev_line, "") != 0) {
                    fprintf(sigma_file, "%s", prev_line);
                }
        
                strcpy(prev_line, line);
            }
        
            modifyLastLine(sigma_last, rho_last, line, sigma, rho);
            fprintf(sigma_file, "%s", sigma_last);
        }
        if (strcmp(rho_name,file_name)!=0 ){
            FILE *rho_file;
            rho_file = fopen(rho_name, "w");
        
            char line[LEN], prev_line[LEN]="", sigma_last[LEN+20], rho_last[LEN+20];
        
            if(file == NULL) { printf("Not able to open the file."); exit(0); }
        
            while(fgets(line, LEN, file) !=NULL){
        
                if (strcmp(prev_line, "") != 0) {
                    fprintf(rho_file, "%s", prev_line);
                }
        
                strcpy(prev_line, line);
            }
        
            modifyLastLine(sigma_last, rho_last, line, sigma, rho);
            fprintf(rho_file, "%s", rho_last);
        }

    }

    FILE *names_file;
    names_file = fopen("eq_names.txt", "w");

    fprintf(names_file, "EQ = %s\n", file_name);
    fprintf(names_file, "EQ_SIGMA = %s\n", sigma_name);
    fprintf(names_file, "EQ_RHO = %s\n", rho_name);

}

int main(int argc, char **argv){

    if(argc>1){
        createSigmaRho(argv[1]);
    } else {
        printf("You must specify an .eq file.\n");
    }

    return 0;
}
