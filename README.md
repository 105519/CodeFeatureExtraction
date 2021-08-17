# 2.0 update

Data&code see folder CFEv2.0

## How to run 'structure graph code BERT'

### quick run

```
cd CFEv2.0/py150_files
unzip washed_python150k.zip
cd ../src
jupyter notebook (run SGC-BERT.ipynb)
```

### full run

```
cd CFEv2.0/py150_files
unzip data.zip
cd ../src/parser
bash build.sh
cd ..
jupyter notebook (run data_wash&ENRE.ipynb)
jupyter notebook (run data_prepare.ipynb)
jupyter notebook (run SGC-BERT.ipynb)
```

### run baseline(quick run)

```
cd CFEv2.0/py150_files
unzip washed_python150k.zip
cd ../src
jupyter notebook (run CODE-BERT.ipynb)
```

### run baseline(full run)

```
cd CFEv2.0/py150_files
unzip data.zip
cd ../src/parser
bash build.sh
cd ..
jupyter notebook (run data_wash&ENRE.ipynb)
jupyter notebook (run data_prepare.ipynb)
jupyter notebook (run CODE-BERT.ipynb)
```

# Code Feature Extraction

## Data Process

In folder *data*, there are two data processing code *ENRE.py* and *DataProcess.py* and few data in folder *py150_files* for playing a demo.

In folder *py150_files*, there is another README file that introduces the data set.

You can play the demo using the following command:

```
cd data
python ENRE.py
python DataProcess.py
```

Then you can see the processed data in folder *py150_new*.

## Tools

In folder *tools*, there are three code dependency extraction tools obtained from internet and a survey document which briefly describe these tools.

