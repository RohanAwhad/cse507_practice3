# Practice 3

Train SWIN for instance segmentation task.

- Dataset: !curl -O "http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/segmentation02.zip"
    - It has been downloaded, and unzipped at `./data/segmentation02`
    - Files (example I haven't added all files, read and manipulate them using python):
        data/
        └── segmentation02/
            └── segmentation02/
                └── segmentation/
                    ├── label_test/
                    │   ├── case227_label.png
                    │   ├── case233_label.png
                    │   ├── case212_label.png
                    ├── org_test/
                    │   ├── case243.bmp
                    │   ├── case242.bmp
                    │   ├── case240.bmp
                    ├── result/
                    ├── org_train/
                    │   ├── case109.bmp
                    │   ├── case135.bmp
                    │   ├── case121.bmp
                    ├── label_train_s/
                    │   ├── case011_label.png
                    │   ├── case005_label.png
                    │   ├── case024_label.png
                    ├── org/
                    │   ├── case109.bmp
                    │   ├── case135.bmp
                    │   ├── case121.bmp
                    ├── label/
                    │   ├── case157_label.png
                    │   ├── case227_label.png
                    │   ├── case011_label.png
                    ├── org_train_s/
                    │   ├── case022.bmp
                    │   ├── case023.bmp
                    │   ├── case009.bmp
                    └── label_train/
                        ├── case157_label.png
                        ├── case011_label.png
                        ├── case005_label.png
        - There are a lot more files, but these are all the dirs and all the filetypes
    ```python
    # Define the color mapping
    color_mapping = {
        255: 'lung',
        85: 'heart',
        170: 'outside_lung',
        0: 'outside_body'
    }
    ```
- Model: facebook/mask2former-swin-small-coco-instance
- Libraries: Transformers and PyTorch


- Train using custom train function. Pass the model and dataset and everything into it. Create a Torch Dataset and dataloader.
- Handle data preprocessing
- You will also need to create a validation loop.
- Compute loss appropriately
- Use AdamW as optimizer

---
Always think step-by-step. Search, code, execute, iterate, repeat till you get the code running for 10 images. 8 for train 2 val. Epochs 2, lr=3e-4. Validate before each epoch and after full-training
Use pretrained weights wherever possible. Write small code blocks. remember you are writing in jupyter notebook style env. previous code will be in memory, and so will their output.
