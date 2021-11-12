# Karyogram-Extraction

A script to extract chromosomes from karyograms and save them individually or as chromosome pairs. It is currently limited to a normal amount of chromosomes, i.e. 46. It only works on standard karyograms that contain 4 rows of chromosomes, whereas the rows contain chromosome 1-7, 6-12, 13-18, 19-X/Y respectively.



## Usage

Three folders are needed: A source folder where the karyograms are located, a folder where the extracted chromosomes are going to be saved, and a fail folder that saves intermediate representations of the algorithm for failed extractions, which can be used to adapt the parameters or spot a genome with mutations. Call it like this: 

```shell
python3 karyogram_extraction/extract.py -s ./imgs/ -d ./extracted/ -f ./fails
```



#### Arguments:

**-s:** Path to image source folder.

**-d:** Destination folder for extracted chromosomes.

**-f:** Folder where failed images will be saved.

**-pair:** True, if chromosome pairs are supposed to be extracted. False for single chromosomes.

**-min_area:** The minimum pixel area a chromosome is supposed to have. Good for removing numbers, letters etc. Standard: a chromosome has to occupy $0.003\%$ of the pixels in an image.


## Example

Here you can see some example for some successful and failed extractions. The images can be found in the  `extracted` and `fails` folder. Example karyograms are from Wikipedia.

### Successful Extraction

#### Karyogram

![karyo1](./imgs/karyo1.jpeg)

#### Extracted chromosomes

**Left chromosome 1**

![1_1_karyo1](./extracted/1_1_karyo1.png)

**Right X chromosome**

![23_2_karyo1](./extracted/23_2_karyo1.png)

### Failed Extraction

An example of a failed extraction. We can see that chromosome three was detected as a single component, indicated by the single + and identical color. Resulting in only 45 single chromosomes being detected. Unfortunately, some of the chromosome are inseparable due to the employed methods here (Connected Component Analysis). The morphological kernel in this case could be adapted, however, this often results in splitting chromosomes elsewhere, which makes it hard to find the optimal set of parameters for every type of chromosome.


## Naming Convention
 Extracted chromosomes are named in the following manner:

**A_B_CCCC.jpg**

  - A is the chromosome number from 1 to 23, where no. 23 is either X or Y chromosome (no differentiation between both is currently done.)
  - B shows encodes if it is the left (=1) or the right (=2) chromosome of the pair. If chromosomes pairs are extracted B=12
  - CCCC the original karyogram file name where the chromosomes were extracted from

 ## Citation
Check out or paper on how to conditionally generate chromosome images based on their banding patterns:
```
@article{uzolas2021deep,
  title={Deep Anomaly Generation: An Image Translation Approach of Synthesizing Abnormal Banded Chromosome Images},
  author={Uzolas, Lukas and Rico, Javier and Coup{\'e}, Pierrick and SanMiguel, Juan C and Cserey, Gy{\"o}rgy},
  journal={arXiv preprint arXiv:2109.09702},
  year={2021}
}

```
