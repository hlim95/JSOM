# JSOM

Repository for the paper "."

![](/images/map1.gif)
![](/images/map2.gif)

⋅ Please run the following for a quick start:
python JSOM.py --file1 'path_to_data1' --file2 'path_to_data2' --matching1 'path_to_matching1' --matching2 'path_to_matching2'

⋅ All input files need to be in .csv format, where each row corresponds to a data point and each column to a feature. 
Data1(file1) and Data2(file2) could have different number of columns, but matching1 and matching2 files should have the same number of columns.
If Data1 and Data2 have the same number of columns, and you want to use the original features to automatically calculuate matching1 and 2, you could skip --matching1 and --matching2.

⋅ Please direct any questions to hlim95@gatech.edu
