import pandas as pd
import torch

#Simulates univariate data based on demo mutation rates in original TADA code

def genUnivData(gamma, q, N_family, N_case, N_control, mut_1, mut_2):
    #Rates for poisson distributions that the counts follow

    X_d1_rates = 2 * mut_1 * gamma * N_family
    X_d2_rates = 2 * mut_2 * gamma * N_family
    X_ctrl_rates = 5000 * [q * N_control]
    X_case_rates = 5000 * [q * gamma * N_case]
    X_t_rates = 5000 * [q * gamma * N_family]
    X_nt_rates = 5000 * [q * N_family]

    #Stacks poisson rates and samples
    cls1 = torch.stack([X_d1_rates,torch.tensor(X_t_rates),torch.tensor(X_nt_rates),torch.tensor(X_case_rates),torch.tensor(X_ctrl_rates)])
    cls2 = torch.stack([X_d2_rates,torch.tensor(X_t_rates),torch.tensor(X_nt_rates),torch.tensor(X_case_rates),torch.tensor(X_ctrl_rates)])
    sample_1 = torch.poisson(cls1)
    sample_2 = torch.poisson(cls2)

    #Combines simulated data with mutation rates into table
    table = torch.transpose(torch.cat((mut_1.unsqueeze(0),sample_1,mut_2.unsqueeze(0),sample_2),dim=0),0,1)
    df = pd.DataFrame(table.numpy())
    df.columns = ['mut.cls1','dn.cls1', 'trans.cls1', 'ntrans.cls1', 'case.cls1', 'ctrl.cls1', 'mut.cls2', 'dn.cls2', 'trans.cls2', 'ntrans.cls2', 'case.cls2', 'ctrl.cls2']
    return df


#Read mutation data
tada_file = "TADA_demo_counts_de-novo_and_inherited.txt"
tada_data = pd.read_table(tada_file)

#Specify the number of families and the number of cases and control samples included in the simulation
N_family = 4500
N_case = 1000
N_ctrl = 3000

#Extract mutation rates
mut_cls1 = torch.tensor(tada_data["mut.cls1"])
mut_cls2 = torch.tensor(tada_data["mut.cls2"])

#Varying gamma and q parameters
g_1,g_2,g_3 = 15, 20, 25
q_1,q_2,q_3 = 5* 10 ** -5, 10**-4, 2*10**-4

#Test gamma_1 and q_1
df_1 = genUnivData(g_1, q_1, N_family, N_case, N_ctrl, mut_cls1, mut_cls2)

#Exports output to text file
df_1.to_csv('testUnivData.txt', index=False, sep='\t')