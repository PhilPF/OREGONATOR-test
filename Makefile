# Compiler
CC = gcc

# Eq files
-include eq_names.txt
-include method.txt
-include c_target.txt

# Compiler flags
CFLAGS = -I/usr/local/CBLAS/include -I/usr/local/LAPACKE/include
LDFLAGS = -L/usr/local/CBLAS -L/usr/local/LAPACKE -L/usr/bin

# Libraries
LIBS = -llapacke -llapack -lcblas -lrefblas -ltmglib -lX11 -lm -lgfortran
BASIC_LIBS = -lm

C_K	=\033[1;30m#BLACK
C_R	=\033[1;31m#BLACK
C_G	=\033[1;32m#GREEN 
C_Y	=\033[1;33m#YELLOW
C_B	=\033[1;34m#BLUE
C_M	=\033[1;35m#MAGENTA
C_C	=\033[1;36m#CYAN
C_W	=\033[1;37m#WHITE
C_E	=\033[0m#END COLOR

# Executable name
TARGET = main

# Eq creator executable name
CREATOR = creator

$(TARGET): create $(CREATOR) $(TARGET).c rk.c rk_sigma.c rk_rho.c taylor.c taylor_sigma.c taylor_rho.c
	$(CC) -g $(CFLAGS) -I./methods/$(METHOD) $(LDFLAGS) -o $(TARGET) $(TARGET).c rk.c taylor.c rk_sigma.c taylor_sigma.c rk_rho.c taylor_rho.c $(LIBS) 

taylor.c: $(CREATOR)
	taylor -header -o taylor.h -name taylor $(EQ)
	taylor -header_name taylor.h -jet -jhelper -name taylor -step -o taylor.c $(EQ)

taylor_sigma.c: $(CREATOR)
	taylor -header -o taylor_sigma.h -name taylor_sigma -jlib jet_1 $(EQ_SIGMA)
	taylor -header_name taylor_sigma.h -jlib jet_1 -jet -name taylor_sigma -step -o taylor_sigma.c $(EQ_SIGMA)

taylor_rho.c: $(CREATOR)
	taylor -header -o taylor_rho.h -name taylor_rho -jlib jet_1 $(EQ_RHO)
	taylor -header_name taylor_rho.h -jlib jet_1 -jet -name taylor_rho -step -o taylor_rho.c $(EQ_RHO)

## make create EQ=vdpol_2_2.eq METHOD=Gauss4
create:
	@echo "METHOD = $(METHOD)" > method.txt
	@$(CC) -I./methods/$(METHOD) -o $(CREATOR) eq_creator.c
	@./$(CREATOR) $(EQ)
	@echo "${C_Y}Creating necessary files to integrate ${C_M}$(EQ)${C_Y} with ${C_M}$(METHOD)${C_Y} method${C_E}"

list_METHOD:
	@echo "The available IRK methods are:"
	@ls -d -1 methods/*/ | sed 's/\///g ; s/methods//'

list_EQ:
	@echo "The available .eq files are:"
	@ls -d -1 *.eq

show:
	@echo "The actual settings are:"
	@echo "EQ=$(EQ), METHOD=$(METHOD)"

# Clean the generated files
clean:
	rm -f $(CREATOR) $(EQ_SIGMA) $(EQ_RHO) eq_names.txt

erk: erk.c taylor.c
	$(CC) -g $(CFLAGS) $(LDFLAGS) -o erk taylor.c erk.c $(LIBS)

_FORCE: ;

OD: _FORCE
	@echo "CODE = $(CODE)\nH_EQ = $(H_EQ)\nI_EQ = $(I_EQ)" > c_target.txt
	@echo 
	@make --no-print-directory EQ=$(H_EQ)
	@echo "\n${C_Y}Running file ${C_M}./OD${C_E}"
	@make compile --no-print-directory CODE=OD
	@./OD
	@echo "\n${C_Y}Running file ${C_M}./OD_PD${C_E}"
	@make compile --no-print-directory CODE=OD_PD
	@./OD_PD
	@make --no-print-directory EQ=$(I_EQ)
	@echo "\n${C_Y}Running file ${C_M}./OD_RHS${C_E}"
	@make compile --no-print-directory CODE=OD_RHS
	@./OD_RHS
	@echo "\n${C_Y}Running file ${C_M}./OD_IPD${C_E}"
	@make compile --no-print-directory CODE=OD_IPD
	@./OD_IPD
	@echo "\n${C_Y}Running file ${C_M}./OD_TEST${C_E}"
	@make compile --no-print-directory CODE=OD_TEST
	@./OD_TEST

compile:
	@echo "CODE = $(CODE)\nH_EQ = $(H_EQ)\nI_EQ = $(I_EQ)" > c_target.txt
	$(CC) $(CFLAGS) -I./methods/$(METHOD) $(LDFLAGS) -o $(CODE) $(CODE).c rk.c taylor.c rk_sigma.c taylor_sigma.c rk_rho.c taylor_rho.c $(LIBS) 

debug:
	@echo "CODE = $(CODE)\nH_EQ = $(H_EQ)\nI_EQ = $(I_EQ)" > c_target.txt
	$(CC) -g -O0 $(CFLAGS) -I./methods/$(METHOD) $(LDFLAGS) -o $(CODE) $(CODE).c rk.c taylor.c rk_sigma.c taylor_sigma.c rk_rho.c taylor_rho.c $(LIBS) 