"""
Date: 9th September 2024
Author: Egheosa Ogbomo

Script to Convert IUPAC Names to SMILES
"""

from rdkit import Chem
from rdkit.Chem import AllChem

from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
import sys
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/eeo21/VSCodeProjects/MLForTribology/Analysis/EBDatasetSMILES.csv')
data['Type'] = data['Type'].str.strip()

# Filter out rows where the 'Type' is 'Amines' or 'Alcohols and Ethers'
filtered_data = data[~data['Type'].isin(['Amine', 'Alcohols and Ethers'])]

plt.figure(figsize=(10, 6))


# Create scatter plot using 'tab10' colormap
scatter = plt.scatter(filtered_data['C-Number'], filtered_data['Experimental_40C_Viscosity'], 
                      c=pd.factorize(filtered_data['Type'])[0], cmap='tab10', label=filtered_data['Type'])

# Add labels and title
plt.xlabel('C-Number')
plt.ylabel('Experimental Viscosity at 40°C')
plt.title('C-Number vs Experimental Viscosity at 40°C (Excluding Amines and Alcohols & Ethers)')

# Add legend
legend_labels = dict(enumerate(pd.factorize(filtered_data['Type'])[1]))
handles, _ = scatter.legend_elements()
plt.legend(handles=handles, labels=[legend_labels[i] for i in range(len(legend_labels))], title="Type")

# Show plot
plt.grid(True)
plt.savefig('40C_Composition.png')
plt.show()


plt.figure(figsize=(10, 6))


# Create scatter plot using 'tab10' colormap
scatter = plt.scatter(filtered_data['C-Number'], filtered_data['Experimental_100C_Viscosity'], 
                      c=pd.factorize(filtered_data['Type'])[0], cmap='tab10', label=filtered_data['Type'])

# Add labels and title
plt.xlabel('C-Number')
plt.ylabel('Experimental Viscosity at 100°C')
plt.title('C-Number vs Experimental Viscosity at 100°C (Excluding Amines and Alcohols & Ethers)')

# Add legend
legend_labels = dict(enumerate(pd.factorize(filtered_data['Type'])[1]))
handles, _ = scatter.legend_elements()
plt.legend(handles=handles, labels=[legend_labels[i] for i in range(len(legend_labels))], title="Type")

# Show plot
plt.grid(True)
plt.savefig('100C_Composition.png')
plt.show()

c_number_counts = data['C-Number'].value_counts().sort_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(c_number_counts.index, c_number_counts.values, color='blue')

# Add labels and title
plt.xlabel('C-Number')
plt.ylabel('Density (Number of Entries)')
plt.title('Density of Entries at Each C-Number')

# Show plot
plt.grid(True)
plt.savefig('Range.png')
plt.show()



# def CIRconvert(ids):
#     try:
#         url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
#         ans = urlopen(url).read().decode('utf8')
#         return ans
#     except:
#         return 'Did not work'

# SMILESList = []

# for ids in identifiers :
#     SMILES = CIRconvert(ids)
#     print(ids, SMILES)
#     SMILESList.append(CIRconvert(ids))

# print(SMILESList)

# EBData['SMILES'] = SMILESList

# EBData.to_csv('C:/Users/eeo21/VSCodeProjects/MLForTribology/Analysis/EBDatasetSMILES.csv')
