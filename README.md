# JSOM

### Repository for the paper "."

**JSOM(Jointly Evolving Self-organzing Maps)** is an algorithm aims to align two related datasets that contain similar clusters by constructing two maps—low dimensional discretized representation of datasets–that jointly evolve according to both datasets. Below, it shows how two maps evolve over time when traiend with two independent MNIST datasets. 

![](/images/map1.gif)
![](/images/map2.gif)

JSOM was specifically tested with various biologcial datasets including data acquried from flow cytometry, mass cytometry and single-cell RNA sequencing, as demonstrated below. Please refer to the paper for more information about JSOM and its results. 



*Please run the following for a quick start:
python JSOM.py --file1 'path_to_data1' --file2 'path_to_data2' --matching1 'path_to_matching1' --matching2 'path_to_matching2'*

⋅ All input files need to be in .csv format, where each row corresponds to a data point and each column to a feature. 
⋅ Data1(file1) and Data2(file2) could have different number of columns, but matching1 and matching2 files should have the same number of columns.
⋅ If Data1 and Data2 have the same number of columns, and you want to use the original features to automatically calculuate matching1 and 2, you could skip --matching1 and --matching2.

*Please direct any questions to hlim95@gatech.edu*
