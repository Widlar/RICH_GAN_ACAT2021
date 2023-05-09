# Reproducing plots from paper 

First, install all dependencies with the command:
```bash
pip install -r /path/to/requirements.txt
```
On a computer without an Nvidia GPU, 
place train and test data in the dataset folder and run: 

```bash
run_single_job.py --config=richgan/configs/simple.mc/simple.mc.muon.config.yaml \ 
--gpu_num=-1 \ 
--schema=evaluation \
--no_uuid_suffix
```
The graphs will appear in the 
```
logs/SimpleModelMuonMC/eval/pdf/004999
```
in pdf format, 
then they can be merged using the files:  
```
DLLs.tex
efficiency_ratio.tex
efficiency_ratio2.tex 
```

# Training Cramer GAN

```bash
python run_single_job.py --config=richgan/configs/simple.mc/simple.mc.muon.config.yaml \ 
--gpu_num=0 \ 
--schema=training \ 
--no_uuid_suffix
```